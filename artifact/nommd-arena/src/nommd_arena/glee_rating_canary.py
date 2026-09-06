"""Causal Bargaining rating-shadow registration and maturation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import statistics
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .glee_activity_eda import _file_digest
from .glee_joint_rating_analysis import _terminal_view, eta_for_game_count, predict_display_delta
from .glee_joint_rating_v2_analysis import empirical_midrank, shrinkage_weight
from .glee_negotiation_rating_v2_4 import RidgeRatingModel
from .glee_statistical_overlay import LIVE_STATISTICAL_OVERLAY_CONTRACT
from .glee_statistical_package import STATISTICAL_PACKAGE_CONTRACT


RATING_CANARY_CONTRACT = "glee-bargaining-rating-canary-v1"
RATING_CANARY_REGISTRY_CONTRACT = "glee-bargaining-rating-canary-registry-v1"
RATING_CANARY_SEED_KIND = "glee-bargaining-rating-canary-seed"
EXPECTED_RATING_MANIFEST_SHA256 = "358c25d2969f07ee1988d0ac4be35aad581c7a37a40cc8be7b3e0ec2fe968d71"
RATING_CUTOFF = "2026-08-12T18:31:08.569568+00:00"
EPOCH_ORIGIN = "2026-08-10T19:10:36.551149+00:00"
REPORTER_CONTRACT = "glee-arena-reporter-v1"
REPORTER_KIND = "glee-arena-family-frontier"
HISTORY_CONTRACT = "glee-rating-history-v1"
RATING_CURVE_SHARES = (0.05, 0.1, 0.2, 0.25, 1 / 3, 0.4, 0.45, 0.5, 0.55, 0.6, 2 / 3, 0.75, 0.8, 0.9, 0.95)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _other_player(player: str) -> str:
    if player == "player_1":
        return "player_2"
    if player == "player_2":
        return "player_1"
    raise ValueError(f"unsupported player identity: {player}")


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _package_receipt(reader: object) -> dict[str, object]:
    receipt = getattr(reader, "receipt", None)
    if not isinstance(receipt, Mapping) or receipt.get("contract") != STATISTICAL_PACKAGE_CONTRACT:
        raise RuntimeError("rating canary requires a verified opponent statistical-package base")
    if receipt.get("live_overlay_contract") != LIVE_STATISTICAL_OVERLAY_CONTRACT:
        raise RuntimeError("rating canary requires the transactional statistical-package live overlay")
    status_method = getattr(reader, "status", None)
    if not callable(status_method):
        raise RuntimeError("rating canary statistical-package reader has no live status interface")
    status = status_method()
    if not isinstance(status, Mapping) or status.get("contract") != LIVE_STATISTICAL_OVERLAY_CONTRACT:
        raise RuntimeError("rating canary statistical-package live overlay is not active")
    if not isinstance(status.get("revision"), int) or not isinstance(status.get("state_sha256"), str):
        raise RuntimeError("rating canary statistical-package live overlay has no durable frontier")
    return {**dict(receipt), "live_overlay_status": dict(status)}


def validate_package_context(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or value.get("contract") != STATISTICAL_PACKAGE_CONTRACT:
        raise RuntimeError("rating canary turn lacks its statistical-package context")
    overlay = value.get("live_overlay")
    if not isinstance(overlay, Mapping) or overlay.get("contract") != LIVE_STATISTICAL_OVERLAY_CONTRACT:
        raise RuntimeError("rating canary turn lacks its statistical-package live overlay")
    if not isinstance(overlay.get("revision"), int) or not isinstance(overlay.get("state_sha256"), str):
        raise RuntimeError("rating canary turn has an incomplete statistical-package overlay receipt")
    return deepcopy(dict(value))


def seal_rating_canary_seed(*, model_dir: Path, protocol_path: Path, seed_path: Path) -> dict[str, object]:
    """Extract the executable frozen Bargaining model into a portable, self-hashed seed."""
    model_dir = model_dir.resolve()
    protocol_path = protocol_path.resolve()
    manifest_path = model_dir / "manifest.json"
    if _file_digest(manifest_path) != EXPECTED_RATING_MANIFEST_SHA256:
        raise RuntimeError("rating-canary source manifest differs from the frozen model")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_receipt = manifest.get("artifacts", {}).get("hybrid-model.json")
    if not isinstance(model_receipt, Mapping) or _file_digest(model_dir / "hybrid-model.json") != model_receipt.get("sha256"):
        raise RuntimeError("rating-canary source model hash mismatch")
    artifact = json.loads((model_dir / "hybrid-model.json").read_text(encoding="utf-8"))
    prospective = artifact.get("prospective_shadow_models", {}).get("bargaining")
    intervals = artifact.get("evaluation_models", {}).get("bargaining", {}).get("intervals")
    if artifact.get("contract") != "glee-joint-rating-reconstruction-v2" or not isinstance(prospective, Mapping) or not isinstance(intervals, Mapping):
        raise RuntimeError("rating-canary source omits its Bargaining prospective model")
    if prospective.get("latest_completed_at") != RATING_CUTOFF or manifest.get("latest_admitted_completion") != RATING_CUTOFF:
        raise RuntimeError("rating-canary source cutoff differs from the protocol")
    implementation_path = Path(__file__).resolve()
    seed: dict[str, object] = {
        "contract": RATING_CANARY_CONTRACT,
        "schema_version": 1,
        "kind": RATING_CANARY_SEED_KIND,
        "rating_cutoff": RATING_CUTOFF,
        "epoch_origin": EPOCH_ORIGIN,
        "source": {
            "contract": manifest["contract"],
            "frontier_sequence": manifest["frontier_sequence"],
            "manifest_sha256": EXPECTED_RATING_MANIFEST_SHA256,
            "hybrid_model_sha256": model_receipt["sha256"],
        },
        "eta_schedule": deepcopy(artifact["eta_schedule"]),
        "bargaining_model": deepcopy(dict(prospective)),
        "intervals": deepcopy(dict(intervals)),
        "implementation_sha256": {
            implementation_path.name: _file_digest(implementation_path),
            protocol_path.name: _file_digest(protocol_path),
        },
        "boundary": "Prospective shadow only; this seed cannot alter prompts, actions, timing, matchmaking, identities, or ratings.",
    }
    seed["seed_sha256"] = _sha(seed)
    _atomic_json(seed_path.resolve(), seed)
    return {"contract": RATING_CANARY_CONTRACT, "seed_path": str(seed_path.resolve()), "seed_sha256": seed["seed_sha256"], "bytes": seed_path.resolve().stat().st_size}


def load_rating_canary_seed(path: Path, *, protocol_path: Path | None = None, verify_implementation: bool = True) -> dict[str, object]:
    seed = json.loads(path.read_text(encoding="utf-8"))
    expected = str(seed.get("seed_sha256") or "")
    unsigned = dict(seed)
    unsigned.pop("seed_sha256", None)
    if _sha(unsigned) != expected:
        raise RuntimeError("rating-canary seed hash mismatch")
    if seed.get("contract") != RATING_CANARY_CONTRACT or seed.get("kind") != RATING_CANARY_SEED_KIND or seed.get("schema_version") != 1:
        raise ValueError("unsupported rating-canary seed")
    if seed.get("rating_cutoff") != RATING_CUTOFF or seed.get("epoch_origin") != EPOCH_ORIGIN:
        raise RuntimeError("rating-canary seed boundary differs from the implementation")
    if verify_implementation:
        selected_protocol = protocol_path or Path(__file__).resolve().parents[2] / "protocols" / "glee-bargaining-rating-canary-v1.md"
        actual = {Path(__file__).name: _file_digest(Path(__file__)), selected_protocol.name: _file_digest(selected_protocol)}
        if seed.get("implementation_sha256") != actual:
            raise RuntimeError("rating-canary implementation differs from its sealed seed")
    return seed


class PublicRatingReader:
    """Read one fresh collision-free public leaderboard row from the reporter frontier."""

    def __init__(self, root: Path, *, max_age_s: float = 30.0) -> None:
        if max_age_s <= 0:
            raise ValueError("public-rating maximum age must be positive")
        self.root = root.resolve()
        self.current_path = self.root / "current.json"
        self.max_age_s = max_age_s

    def player(self, family: str, player_id: str, *, now: str | None = None) -> dict[str, object]:
        if not self.current_path.is_file():
            return {"status": "unavailable", "reason": "reporter-frontier-missing"}
        frontier = json.loads(self.current_path.read_text(encoding="utf-8"))
        if frontier.get("contract") != REPORTER_CONTRACT or frontier.get("kind") != REPORTER_KIND or frontier.get("schema_version") != 2:
            raise RuntimeError("unsupported public reporter frontier")
        expected = frontier.get("frontier_sha256")
        actual = _sha({key: value for key, value in frontier.items() if key != "frontier_sha256"})
        if expected != actual:
            raise RuntimeError("public reporter frontier hash mismatch")
        observed_at = str(frontier["completed_at"])
        reference = _timestamp(now or _now())
        age_s = max(0.0, reference - _timestamp(observed_at))
        if age_s > self.max_age_s:
            return {"status": "unavailable", "reason": "reporter-frontier-stale", "age_s": age_s, "sequence": frontier.get("sequence"), "observed_at": observed_at}
        family_frontier = frontier.get("families", {}).get(family)
        if not isinstance(family_frontier, Mapping) or family_frontier.get("poll", {}).get("status") != "ok":
            return {"status": "unavailable", "reason": "family-poll-unavailable", "sequence": frontier.get("sequence"), "observed_at": observed_at}
        sequence = int(frontier["sequence"])
        rows = [entry for entry in family_frontier.get("rows", []) if isinstance(entry, Mapping) and entry.get("row", {}).get("player_id") == player_id and entry.get("last_observed_sequence") == sequence]
        if len(rows) != 1:
            return {"status": "unavailable", "reason": "public-id-absent-or-ambiguous", "matches": len(rows), "sequence": sequence, "observed_at": observed_at}
        row = rows[0]["row"]
        rating = _finite(row.get("rating"))
        games_played = row.get("games_played")
        if rating is None or isinstance(games_played, bool) or not isinstance(games_played, int) or games_played < 1:
            return {"status": "unavailable", "reason": "public-row-missing-rating-state", "sequence": sequence, "observed_at": observed_at}
        return {
            "status": "available",
            "source": "fresh-public-reporter",
            "public_player_id": player_id,
            "player_name": row.get("player_name"),
            "rating": rating,
            "games_played": games_played,
            "sequence": sequence,
            "observed_at": observed_at,
            "age_s": age_s,
            "row_sha256": _sha(row),
        }


class BargainingRatingPredictor:
    """Evaluate the frozen exact-configuration hybrid from either player's terminal perspective."""

    def __init__(self, seed: Mapping[str, object]) -> None:
        model = seed.get("bargaining_model")
        intervals = seed.get("intervals")
        if not isinstance(model, Mapping) or not isinstance(intervals, Mapping):
            raise ValueError("rating-canary seed lacks its Bargaining model")
        self.seed_sha256 = str(seed["seed_sha256"])
        self.rating_cutoff = str(seed["rating_cutoff"])
        self.epoch_origin = str(seed["epoch_origin"])
        self.eta_schedule = deepcopy(dict(seed["eta_schedule"]))
        self.model = RidgeRatingModel.from_dict(model["structural_model"])
        self.references = {str(key): tuple(float(value) for value in values) for key, values in dict(model["references"]).items()}
        self.residuals = {str(key): dict(value) for key, value in dict(model["configuration_residuals"]).items()}
        self.rank_alpha = float(model["rank_alpha"])
        self.residual_alpha = float(model["residual_alpha"])
        self.intervals = {str(key): float(value) for key, value in intervals.items()}
        self.training_samples = int(model["training_samples"])

    def manifest_receipt(self) -> dict[str, object]:
        return {"contract": RATING_CANARY_CONTRACT, "seed_sha256": self.seed_sha256, "rating_cutoff": self.rating_cutoff, "training_samples": self.training_samples, "configuration_references": len(self.references), "configuration_residuals": len(self.residuals)}

    def predict(self, terminal: Mapping[str, object], *, target_player: str, target_rating: float, target_games: int, other_rating: float | None, terminal_at: str) -> dict[str, object]:
        view = _terminal_view(terminal, target_player)
        features = {str(key): float(value) for key, value in dict(view["base_features"]).items()}
        features["completion_epoch_days"] = (_timestamp(terminal_at) - _timestamp(self.epoch_origin)) / 86400.0
        features["opponent_pregame_rating_known"] = float(other_rating is not None)
        features["opponent_pregame_rating_scaled"] = (float(other_rating) - 2000.0) / 1000.0 if other_rating is not None else 0.0
        structural_percentile = self.model.predict(features)
        configuration = str(view["observed_configuration_sha256"])
        empirical, rank_support = empirical_midrank(float(view["own_payoff"]), self.references.get(configuration, ())) if configuration in self.references else (None, 0)
        rank_weight = shrinkage_weight(rank_support, self.rank_alpha)
        hybrid_percentile = structural_percentile if empirical is None else (1.0 - rank_weight) * structural_percentile + rank_weight * empirical
        eta = eta_for_game_count(self.eta_schedule, target_games)
        rank_delta = predict_display_delta(target_rating, target_games, hybrid_percentile, eta)
        residual = self.residuals.get(configuration)
        residual_support = int(residual["count"]) if residual is not None else 0
        residual_correction = shrinkage_weight(residual_support, self.residual_alpha) * float(residual["mean"]) if residual is not None else 0.0
        point = rank_delta + residual_correction
        return {
            "status": "available",
            "target_player": target_player,
            "target_rating": target_rating,
            "target_games": target_games,
            "other_rating_known": other_rating is not None,
            "own_payoff": view["own_payoff"],
            "opponent_payoff": view["opponent_payoff"],
            "configuration_sha256": configuration,
            "structural_percentile": structural_percentile,
            "empirical_percentile": empirical,
            "rank_weight": rank_weight,
            "rank_support": rank_support,
            "hybrid_percentile": hybrid_percentile,
            "rank_delta": rank_delta,
            "residual_support": residual_support,
            "residual_correction": residual_correction,
            "predicted_delta": point,
            "interval_80": [point + self.intervals["lower_80"], point + self.intervals["upper_80"]],
            "interval_95": [point + self.intervals["lower_95"], point + self.intervals["upper_95"]],
            "eta": eta,
        }


class BargainingRatingCanaryRegistry:
    """Store immutable game contexts, shadows, forecasts, targets, and failures."""

    def __init__(self, path: Path, *, metadata: Mapping[str, object]) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._metadata = {str(key): _canonical(value) if not isinstance(value, str) else value for key, value in metadata.items()}
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS game_contexts (game_id TEXT PRIMARY KEY, observed_at TEXT NOT NULL, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, registered_at TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS turn_shadows (turn_key TEXT PRIMARY KEY, game_id TEXT NOT NULL, stage TEXT NOT NULL, causal_observation_at TEXT NOT NULL, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, registered_at TEXT NOT NULL, FOREIGN KEY (game_id) REFERENCES game_contexts(game_id));
                    CREATE TABLE IF NOT EXISTS terminal_forecasts (game_id TEXT PRIMARY KEY, terminal_at TEXT NOT NULL, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, registered_at TEXT NOT NULL, FOREIGN KEY (game_id) REFERENCES game_contexts(game_id));
                    CREATE TABLE IF NOT EXISTS maturations (game_id TEXT NOT NULL, target_scope TEXT NOT NULL, observed_at TEXT NOT NULL, target_json TEXT NOT NULL, target_sha256 TEXT NOT NULL, matured_at TEXT NOT NULL, PRIMARY KEY (game_id, target_scope), FOREIGN KEY (game_id) REFERENCES terminal_forecasts(game_id));
                    CREATE TABLE IF NOT EXISTS failures (failure_id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL, stage TEXT NOT NULL, game_id TEXT, turn_key TEXT, error_type TEXT NOT NULL, error TEXT NOT NULL);
                    CREATE TRIGGER IF NOT EXISTS game_contexts_no_update BEFORE UPDATE ON game_contexts BEGIN SELECT RAISE(ABORT, 'game contexts are append-only'); END;
                    CREATE TRIGGER IF NOT EXISTS game_contexts_no_delete BEFORE DELETE ON game_contexts BEGIN SELECT RAISE(ABORT, 'game contexts are append-only'); END;
                    CREATE TRIGGER IF NOT EXISTS turn_shadows_no_update BEFORE UPDATE ON turn_shadows BEGIN SELECT RAISE(ABORT, 'turn shadows are append-only'); END;
                    CREATE TRIGGER IF NOT EXISTS turn_shadows_no_delete BEFORE DELETE ON turn_shadows BEGIN SELECT RAISE(ABORT, 'turn shadows are append-only'); END;
                    CREATE TRIGGER IF NOT EXISTS terminal_forecasts_no_update BEFORE UPDATE ON terminal_forecasts BEGIN SELECT RAISE(ABORT, 'terminal forecasts are append-only'); END;
                    CREATE TRIGGER IF NOT EXISTS terminal_forecasts_no_delete BEFORE DELETE ON terminal_forecasts BEGIN SELECT RAISE(ABORT, 'terminal forecasts are append-only'); END;
                    CREATE TRIGGER IF NOT EXISTS maturations_no_update BEFORE UPDATE ON maturations BEGIN SELECT RAISE(ABORT, 'maturations are append-only'); END;
                    CREATE TRIGGER IF NOT EXISTS maturations_no_delete BEFORE DELETE ON maturations BEGIN SELECT RAISE(ABORT, 'maturations are append-only'); END;
                    """
                )
                expected = {"contract": RATING_CANARY_REGISTRY_CONTRACT, "schema_version": "1", **self._metadata}
                for key, value in expected.items():
                    existing = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
                    if existing is None:
                        connection.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", (key, value))
                    elif str(existing["value"]) != str(value):
                        raise RuntimeError(f"rating-canary registry metadata conflict: {key}")
                connection.commit()
            finally:
                connection.close()
        self.path.chmod(0o600)

    def _insert_immutable(self, table: str, key_where: str, key_values: tuple[object, ...], columns: Sequence[str], values: Sequence[object]) -> bool:
        with self._lock:
            connection = self._connect()
            try:
                existing = connection.execute(f"SELECT * FROM {table} WHERE {key_where}", key_values).fetchone()
                if existing is not None:
                    comparable = {column: str(existing[column]) for column in columns if column not in {"registered_at", "matured_at"}}
                    proposed = {column: str(value) for column, value in zip(columns, values, strict=True) if column not in {"registered_at", "matured_at"}}
                    if comparable == proposed:
                        return False
                    raise RuntimeError(f"conflicting immutable rating-canary record: {table}/{key_values!r}")
                placeholders = ",".join("?" for _column in columns)
                connection.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})", tuple(values))
                connection.commit()
                return True
            finally:
                connection.close()

    def game_context(self, game_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload_json FROM game_contexts WHERE game_id = ?", (game_id,)).fetchone()
        return json.loads(str(row["payload_json"])) if row is not None else None

    def register_game(self, game_id: str, *, observed_at: str, payload: Mapping[str, object], registered_at: str | None = None) -> bool:
        canonical = _canonical(payload)
        values = (game_id, observed_at, canonical, hashlib.sha256(canonical.encode()).hexdigest(), registered_at or _now())
        return self._insert_immutable("game_contexts", "game_id = ?", (game_id,), ("game_id", "observed_at", "payload_json", "payload_sha256", "registered_at"), values)

    def register_turn(self, turn_key: str, *, game_id: str, stage: str, causal_observation_at: str, payload: Mapping[str, object], registered_at: str | None = None) -> bool:
        canonical = _canonical(payload)
        values = (turn_key, game_id, stage, causal_observation_at, canonical, hashlib.sha256(canonical.encode()).hexdigest(), registered_at or _now())
        return self._insert_immutable("turn_shadows", "turn_key = ?", (turn_key,), ("turn_key", "game_id", "stage", "causal_observation_at", "payload_json", "payload_sha256", "registered_at"), values)

    def register_terminal(self, game_id: str, *, terminal_at: str, payload: Mapping[str, object], registered_at: str | None = None) -> bool:
        if _timestamp(terminal_at) <= _timestamp(RATING_CUTOFF):
            raise ValueError("rating-canary terminal is not after the frozen cutoff")
        canonical = _canonical(payload)
        values = (game_id, terminal_at, canonical, hashlib.sha256(canonical.encode()).hexdigest(), registered_at or _now())
        return self._insert_immutable("terminal_forecasts", "game_id = ?", (game_id,), ("game_id", "terminal_at", "payload_json", "payload_sha256", "registered_at"), values)

    def mature(self, game_id: str, *, target_scope: str, observed_at: str, target: Mapping[str, object], matured_at: str | None = None) -> bool:
        if target_scope not in {"self", "opponent"}:
            raise ValueError("rating-canary target scope must be self or opponent")
        with self._connect() as connection:
            registration = connection.execute("SELECT registered_at FROM terminal_forecasts WHERE game_id = ?", (game_id,)).fetchone()
        if registration is None:
            raise KeyError("rating-canary target has no terminal forecast")
        if _timestamp(observed_at) <= _timestamp(str(registration["registered_at"])):
            raise ValueError("rating-canary target was observable before or at registration")
        canonical = _canonical(target)
        values = (game_id, target_scope, observed_at, canonical, hashlib.sha256(canonical.encode()).hexdigest(), matured_at or _now())
        return self._insert_immutable("maturations", "game_id = ? AND target_scope = ?", (game_id, target_scope), ("game_id", "target_scope", "observed_at", "target_json", "target_sha256", "matured_at"), values)

    def failure(self, *, stage: str, error: BaseException, game_id: str | None = None, turn_key: str | None = None) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("INSERT INTO failures(occurred_at, stage, game_id, turn_key, error_type, error) VALUES (?, ?, ?, ?, ?, ?)", (_now(), stage, game_id, turn_key, type(error).__name__, str(error)))
            connection.commit()

    def reconcile_self_history(self, history_path: Path) -> dict[str, object]:
        if not history_path.is_file():
            return {"status": "unavailable", "reason": "rating-history-missing", "matured": 0}
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history.get("contract") != HISTORY_CONTRACT or history.get("schema_version") != 1:
            raise RuntimeError("unsupported authenticated rating-history artifact")
        observed_at = str(history["synchronized_at"])
        deltas = history.get("game_deltas") if isinstance(history.get("game_deltas"), Mapping) else {}
        matured = 0
        with self._connect() as connection:
            rows = connection.execute("SELECT game_id, registered_at FROM terminal_forecasts ORDER BY registered_at, game_id").fetchall()
            existing = {str(row["game_id"]) for row in connection.execute("SELECT game_id FROM maturations WHERE target_scope = 'self'")}
        for row in rows:
            game_id = str(row["game_id"])
            target = deltas.get(game_id)
            if game_id in existing or not isinstance(target, Mapping) or _timestamp(observed_at) <= _timestamp(str(row["registered_at"])):
                continue
            matured += int(self.mature(game_id, target_scope="self", observed_at=observed_at, target={"rating_delta": float(target["rating_delta"]), "completed_at": target.get("completed_at"), "revision": target.get("revision"), "record_sha256": target.get("record_sha256"), "history_game_deltas_sha256": history.get("game_deltas_sha256")}))
        return {"status": "available", "synchronized_at": observed_at, "history_game_count": len(deltas), "matured": matured}

    def status(self) -> dict[str, object]:
        with self._connect() as connection:
            counts = {table: int(connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]) for table in ("game_contexts", "turn_shadows", "terminal_forecasts", "maturations", "failures")}
            self_maturations = int(connection.execute("SELECT COUNT(*) AS count FROM maturations WHERE target_scope = 'self'").fetchone()["count"])
            rows = connection.execute("SELECT f.payload_json, m.target_json FROM terminal_forecasts f JOIN maturations m ON m.game_id = f.game_id AND m.target_scope = 'self'").fetchall()
            turns = connection.execute("SELECT payload_json FROM turn_shadows").fetchall()
        errors = []
        signs = []
        interval_80 = []
        interval_95 = []
        for row in rows:
            forecast = json.loads(str(row["payload_json"])).get("forecasts", {}).get("self", {})
            target = json.loads(str(row["target_json"]))
            if forecast.get("status") != "available":
                continue
            actual = float(target["rating_delta"])
            prediction = float(forecast["predicted_delta"])
            errors.append(actual - prediction)
            signs.append((actual > 0) == (prediction > 0) if actual != 0.0 and prediction != 0.0 else actual == prediction)
            interval_80.append(float(forecast["interval_80"][0]) <= actual <= float(forecast["interval_80"][1]))
            interval_95.append(float(forecast["interval_95"][0]) <= actual <= float(forecast["interval_95"][1]))
        divergence = 0
        comparable = 0
        for row in turns:
            payload = json.loads(str(row["payload_json"]))
            recommendation = payload.get("recommendations") if isinstance(payload.get("recommendations"), Mapping) else {}
            if recommendation.get("rating_opponent_share") is not None and recommendation.get("submitted_opponent_share") is not None:
                comparable += 1
                divergence += int(not math.isclose(float(recommendation["rating_opponent_share"]), float(recommendation["submitted_opponent_share"]), rel_tol=0.0, abs_tol=1e-6))
        metrics = {
            "count": len(errors),
            "mae": statistics.fmean(abs(value) for value in errors) if errors else None,
            "rmse": math.sqrt(statistics.fmean(value * value for value in errors)) if errors else None,
            "sign_accuracy": statistics.fmean(float(value) for value in signs) if signs else None,
            "coverage_80": statistics.fmean(interval_80) if interval_80 else None,
            "coverage_95": statistics.fmean(interval_95) if interval_95 else None,
        }
        return {"contract": RATING_CANARY_REGISTRY_CONTRACT, "path": str(self.path), **counts, "self_maturations": self_maturations, "self_metrics": metrics, "recommendation_comparisons": comparable, "recommendation_divergences": divergence}


class BargainingRatingCanary:
    """Register shadow rating forecasts without changing the live Bargaining policy."""

    def __init__(self, *, seed_path: Path, protocol_path: Path, registry_path: Path, reporter_root: Path, history_path: Path, package_reader: object, reporter_max_age_s: float = 30.0) -> None:
        self.protocol_path = protocol_path.resolve()
        self.seed_path = seed_path.resolve()
        self.seed = load_rating_canary_seed(self.seed_path, protocol_path=self.protocol_path)
        self.predictor = BargainingRatingPredictor(self.seed)
        self.package_receipt = _package_receipt(package_reader)
        self.reporter = PublicRatingReader(reporter_root, max_age_s=reporter_max_age_s)
        self.history_path = history_path.resolve()
        metadata = {
            "seed_sha256": str(self.seed["seed_sha256"]),
            "protocol_sha256": _file_digest(self.protocol_path),
            "package_manifest_sha256": str(self.package_receipt["manifest_sha256"]),
            "package_release": str(self.package_receipt["release"]),
            "overlay_id": str(self.package_receipt["live_overlay_id"]),
        }
        self.registry = BargainingRatingCanaryRegistry(registry_path, metadata=metadata)
        self._last_reconcile = float("-inf")

    def manifest_receipt(self) -> dict[str, object]:
        package = {key: value for key, value in self.package_receipt.items() if key != "live_overlay_status"}
        return {
            **self.predictor.manifest_receipt(),
            "seed_path": str(self.seed_path),
            "protocol_sha256": _file_digest(self.protocol_path),
            "authority": "shadow-only",
            "statistical_package": package,
        }

    @staticmethod
    def _own_rating(sensor_frontier: Mapping[str, object] | None) -> dict[str, object]:
        if not isinstance(sensor_frontier, Mapping):
            return {"status": "unavailable", "reason": "sensor-frontier-unavailable"}
        stats = sensor_frontier.get("stats") if isinstance(sensor_frontier.get("stats"), Mapping) else {}
        score = stats.get("scores", {}).get("bargaining") if isinstance(stats.get("scores"), Mapping) else None
        if not isinstance(score, Mapping) or _finite(score.get("rating")) is None or isinstance(score.get("games_played"), bool) or not isinstance(score.get("games_played"), int):
            return {"status": "unavailable", "reason": "sensor-frontier-missing-bargaining-rating", "sensor_sequence": sensor_frontier.get("sequence")}
        return {
            "status": "available",
            "source": "authenticated-sensor-frontier",
            "rating": float(score["rating"]),
            "games_played": int(score["games_played"]),
            "sensor_sequence": sensor_frontier.get("sequence"),
            "observed_at": sensor_frontier.get("fetched_at"),
            "frontier_sha256": sensor_frontier.get("frontier_sha256"),
            "stats_sha256": _sha(stats),
        }

    def capture_game(self, game: Mapping[str, object], *, package_context: object, sensor_frontier: Mapping[str, object] | None, observed_at: str) -> bool:
        if game.get("game_family") != "bargaining":
            raise ValueError("rating canary received a non-Bargaining game")
        game_id = str(game["game_id"])
        if self.registry.game_context(game_id) is not None:
            return False
        package = validate_package_context(package_context)
        identity = package.get("identity_resolution") if isinstance(package.get("identity_resolution"), Mapping) else {}
        public_id = str(identity.get("public_player_id")) if identity.get("status") == "exact-current-label" and identity.get("public_player_id") else None
        opponent = self.reporter.player("bargaining", public_id, now=observed_at) if public_id is not None else {"status": "unavailable", "reason": "identity-not-exact", "identity_resolution": identity.get("status")}
        own = self._own_rating(sensor_frontier)
        payload = {
            "contract": RATING_CANARY_CONTRACT,
            "game_id": game_id,
            "family": "bargaining",
            "observed_at": observed_at,
            "identity_mode": "hidden" if game.get("opponent", {}).get("type") == "hidden" else "known",
            "your_player": game.get("your_player"),
            "self": own,
            "opponent": opponent,
            "statistical_package": package,
            "game_configuration_sha256": _sha(_terminal_configuration(game)),
            "rating_authority": "shadow-only",
            "sic_for_hidden_identity": "disabled",
        }
        return self.registry.register_game(game_id, observed_at=observed_at, payload=payload)

    def _terminal_predictions(self, terminal: Mapping[str, object], game_context: Mapping[str, object], *, terminal_at: str) -> dict[str, object]:
        your_player = str(terminal.get("your_player") or game_context.get("your_player") or "")
        other_player = _other_player(your_player)
        own = game_context.get("self") if isinstance(game_context.get("self"), Mapping) else {}
        opponent = game_context.get("opponent") if isinstance(game_context.get("opponent"), Mapping) else {}
        own_available = own.get("status") == "available"
        opponent_available = opponent.get("status") == "available"
        if own_available:
            self_prediction = self.predictor.predict(terminal, target_player=your_player, target_rating=float(own["rating"]), target_games=int(own["games_played"]), other_rating=float(opponent["rating"]) if opponent_available else None, terminal_at=terminal_at)
        else:
            self_prediction = {"status": "unavailable", "reason": own.get("reason", "pregame-self-rating-unavailable")}
        if opponent_available:
            opponent_prediction = self.predictor.predict(terminal, target_player=other_player, target_rating=float(opponent["rating"]), target_games=int(opponent["games_played"]), other_rating=float(own["rating"]) if own_available else None, terminal_at=terminal_at)
        else:
            opponent_prediction = {"status": "unavailable", "reason": opponent.get("reason", "pregame-opponent-rating-unavailable")}
        return {"self": self_prediction, "opponent": opponent_prediction}

    def _candidate_surface(self, game: Mapping[str, object], action: Mapping[str, object], advisor_handle: object | None, context: Mapping[str, object], *, terminal_at: str) -> tuple[list[dict[str, object]], dict[str, object]]:
        action_type = str(game.get("valid_actions", {}).get("type") or game.get("phase") or "")
        submitted_share = _opponent_share(game, action)
        candidates: set[float] = set()
        if action_type == "offer":
            candidates.update(float(value) for value in RATING_CURVE_SHARES)
            if submitted_share is not None:
                candidates.add(submitted_share)
            prompt_context = getattr(advisor_handle, "prompt_context", {}) if advisor_handle is not None else {}
            continuation = prompt_context.get("behavioral_continuation") if isinstance(prompt_context, Mapping) and isinstance(prompt_context.get("behavioral_continuation"), Mapping) else {}
            modeled = continuation.get("modeled_offer_policy") if isinstance(continuation.get("modeled_offer_policy"), Mapping) else {}
            if _finite(modeled.get("opponent_share")) is not None:
                candidates.add(float(modeled["opponent_share"]))
        elif str(action.get("decision") or "").casefold() in {"accept", "acceptoffer"}:
            if submitted_share is not None:
                candidates.add(submitted_share)
        rows = []
        for share in sorted(value for value in candidates if 0.0 < value < 1.0):
            behavior: dict[str, object] = {"status": "unavailable", "reason": "bargaining-advisor-unavailable"}
            if advisor_handle is not None:
                try:
                    evaluation = advisor_handle.rollout.evaluate_offer(advisor_handle.context, share)
                    behavior = {"status": "available", **deepcopy(dict(evaluation))}
                except Exception as error:
                    behavior = {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
            try:
                terminal = _accepted_terminal(game, opponent_share=share)
                forecasts = self._terminal_predictions(terminal, context, terminal_at=terminal_at)
            except Exception as error:
                forecasts = {"self": {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}, "opponent": {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}}
            probability = _finite(behavior.get("opponent_accept_probability_conservative"))
            self_delta = _finite(forecasts["self"].get("predicted_delta"))
            rating_score = probability * self_delta if probability is not None and self_delta is not None else None
            rows.append({"opponent_share": share, "submitted": submitted_share is not None and math.isclose(share, submitted_share, rel_tol=0.0, abs_tol=1e-9), "behavior": behavior, "forecasts_if_accepted": forecasts, "myopic_self_rating_score": rating_score})
        eligible = [row for row in rows if row["myopic_self_rating_score"] is not None]
        rating = max(eligible, key=lambda row: (float(row["myopic_self_rating_score"]), -float(row["opponent_share"]))) if eligible else None
        return rows, {
            "submitted_opponent_share": submitted_share,
            "rating_opponent_share": rating["opponent_share"] if rating is not None else None,
            "rating_score": rating["myopic_self_rating_score"] if rating is not None else None,
            "semantics": "acceptance-probability times immediate accepted-branch self rating delta; rejection continuation is zero and the recommendation is diagnostic only",
        }

    def register_turn(self, *, turn_id: str, stage: str, game: Mapping[str, object], action: Mapping[str, object], advisor_handle: object | None, package_context: object, causal_observation_at: str) -> bool:
        game_id = str(game["game_id"])
        context = self.registry.game_context(game_id)
        if context is None:
            raise RuntimeError("rating-canary turn has no immutable pregame context")
        package = validate_package_context(package_context)
        surface, recommendations = self._candidate_surface(game, action, advisor_handle, context, terminal_at=causal_observation_at)
        payload = {
            "contract": RATING_CANARY_CONTRACT,
            "frontier": "registered-before-network-submission",
            "game_id": game_id,
            "turn_id": turn_id,
            "stage": stage,
            "action": deepcopy(dict(action)),
            "action_sha256": _sha(action),
            "statistical_package": package,
            "candidate_surface": surface,
            "recommendations": recommendations,
            "action_changed": False,
            "authority": "shadow-only",
        }
        return self.registry.register_turn(f"{turn_id}:{stage}", game_id=game_id, stage=stage, causal_observation_at=causal_observation_at, payload=payload)

    def register_terminal(self, terminal: Mapping[str, object], *, terminal_at: str) -> bool:
        game_id = str(terminal["game_id"])
        context = self.registry.game_context(game_id)
        if context is None:
            raise RuntimeError("rating-canary terminal has no immutable pregame context")
        forecasts = self._terminal_predictions(terminal, context, terminal_at=terminal_at)
        payload = {
            "contract": RATING_CANARY_CONTRACT,
            "frontier": "registered-at-first-local-terminal-observation-before-rating-history-read",
            "game_id": game_id,
            "terminal_at": terminal_at,
            "terminal_state_sha256": _sha(terminal),
            "pregame_context_sha256": _sha(context),
            "forecasts": forecasts,
            "authority": "shadow-only",
        }
        return self.registry.register_terminal(game_id, terminal_at=terminal_at, payload=payload)

    def reconcile(self, *, force: bool = False) -> dict[str, object]:
        now = datetime.now(timezone.utc).timestamp()
        if not force and now - self._last_reconcile < 10.0:
            return {"status": "throttled", "matured": 0}
        self._last_reconcile = now
        return self.registry.reconcile_self_history(self.history_path)

    def status(self) -> dict[str, object]:
        return {**self.manifest_receipt(), "registry": self.registry.status()}

    def failure(self, *, stage: str, error: BaseException, game_id: str | None = None, turn_key: str | None = None) -> None:
        self.registry.failure(stage=stage, error=error, game_id=game_id, turn_key=turn_key)

    def close(self) -> None:
        return None


def _terminal_configuration(game: Mapping[str, object]) -> dict[str, object]:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    return {
        "your_player": game.get("your_player"),
        "money_to_divide": state.get("money_to_divide"),
        "delta_1": state.get("delta_1"),
        "delta_2": state.get("delta_2"),
        "complete_information": state.get("complete_information"),
        "horizon_known": state.get("horizon_known"),
        "max_rounds": state.get("max_rounds"),
        "messages_allowed": state.get("messages_allowed"),
    }


def _gain(action: Mapping[str, object], player: str) -> float | None:
    aliases = ("player_1_gain", "alice_gain") if player == "player_1" else ("player_2_gain", "bob_gain")
    for key in aliases:
        value = _finite(action.get(key))
        if value is not None:
            return value
    return None


def _opponent_share(game: Mapping[str, object], action: Mapping[str, object]) -> float | None:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    your_player = str(game.get("your_player") or "")
    opponent_player = _other_player(your_player)
    source: Mapping[str, object] = action
    if str(action.get("decision") or "").casefold() in {"accept", "acceptoffer"}:
        source = state.get("last_offer") if isinstance(state.get("last_offer"), Mapping) else {}
    money = _finite(state.get("money_to_divide"))
    gain = _gain(source, opponent_player)
    return gain / money if money not in (None, 0.0) and gain is not None else None


def _accepted_terminal(game: Mapping[str, object], *, opponent_share: float) -> dict[str, object]:
    state = deepcopy(dict(game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}))
    your_player = str(game.get("your_player") or "")
    opponent_player = _other_player(your_player)
    money = _finite(state.get("money_to_divide"))
    round_number = int(state.get("round") or 1)
    if money is None or money <= 0:
        raise ValueError("Bargaining candidate lacks a positive money_to_divide")
    deltas = {player: _finite(state.get("delta_1" if player == "player_1" else "delta_2")) for player in ("player_1", "player_2")}
    discount_exponent = max(0, round_number - 1)
    if discount_exponent > 0 and any(value is None for value in deltas.values()):
        raise ValueError("Bargaining candidate cannot invent a hidden discount factor")
    gains = {opponent_player: money * opponent_share, your_player: money * (1.0 - opponent_share)}
    payoffs = {player: gains[player] if discount_exponent == 0 else gains[player] * float(deltas[player]) ** discount_exponent for player in ("player_1", "player_2")}
    result = {
        "outcome": "agreement",
        "agreed_round": round_number,
        "agreed_player_1_gain": gains["player_1"],
        "agreed_player_2_gain": gains["player_2"],
        "player_1_payoff": payoffs["player_1"],
        "player_2_payoff": payoffs["player_2"],
    }
    state["phase"] = "completed"
    state["result"] = deepcopy(result)
    return {"game_id": game["game_id"], "game_family": "bargaining", "your_player": your_player, "opponent": deepcopy(game.get("opponent")), "game_state": state, "status": "completed", "result": result}
