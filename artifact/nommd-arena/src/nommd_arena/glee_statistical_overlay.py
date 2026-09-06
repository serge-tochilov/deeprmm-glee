"""Apply every completed GLEE game to a transactional statistical-package overlay."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Mapping

from .glee_activity_eda import GLEE_FAMILIES, _file_digest
from .glee_behavior_corpus import extract_behavior_moves
from .glee_statistical_package import (
    OpponentStatisticalPackageReader,
    _action_observations,
    _bargaining_decision_model,
    _canonical,
    _context_rank,
    _evidence_tier,
    _population_distribution,
    _project_distribution,
    _smoothed_distribution,
    _visible_context,
)


LIVE_STATISTICAL_OVERLAY_CONTRACT = "glee-opponent-statistical-live-overlay-v1"
LIVE_STATISTICAL_OVERLAY_SCHEMA_VERSION = 1


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _merged_counts(base: Mapping[str, Mapping[str, int]], overlay: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    merged: dict[str, Counter[str]] = {}
    for source in (base, overlay):
        for condition, outcomes in source.items():
            target = merged.setdefault(str(condition), Counter())
            target.update({str(outcome): int(count) for outcome, count in outcomes.items()})
    return {condition: dict(sorted(outcomes.items())) for condition, outcomes in sorted(merged.items())}


class LiveOpponentStatisticalPackageReader(OpponentStatisticalPackageReader):
    """Read a verified frozen base plus a shared, idempotent per-game SQLite overlay."""

    def __init__(self, root: Path, *, context_limit: int = 8, outcome_limit: int = 5, busy_timeout_s: float = 30.0) -> None:
        super().__init__(root, context_limit=context_limit, outcome_limit=outcome_limit)
        if busy_timeout_s <= 0:
            raise ValueError("statistical-package overlay busy timeout must be positive")
        self.base_receipt = dict(self.receipt)
        manifest_sha256 = str(self.base_receipt["manifest_sha256"])
        overlay_id = f"{self.base_receipt['release']}-{manifest_sha256[:16]}"
        self.overlay_path = self.root / "live" / f"{overlay_id}.sqlite3"
        self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.overlay_path, timeout=busy_timeout_s, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(f"PRAGMA busy_timeout = {round(busy_timeout_s * 1000)}")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._initialize_overlay()
        self.receipt = {
            **self.base_receipt,
            "live_overlay_contract": LIVE_STATISTICAL_OVERLAY_CONTRACT,
            "live_overlay_schema_version": LIVE_STATISTICAL_OVERLAY_SCHEMA_VERSION,
            "live_overlay_id": overlay_id,
            "live_overlay_implementation_sha256": _file_digest(Path(__file__)),
        }

    def _initialize_overlay(self) -> None:
        expected = {
            "contract": LIVE_STATISTICAL_OVERLAY_CONTRACT,
            "schema_version": str(LIVE_STATISTICAL_OVERLAY_SCHEMA_VERSION),
            "base_manifest_sha256": str(self.base_receipt["manifest_sha256"]),
            "base_release": str(self.base_receipt["release"]),
        }
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS games (family TEXT NOT NULL, game_id TEXT NOT NULL, completed_at TEXT NOT NULL, completion_order INTEGER NOT NULL, evidence_sha256 TEXT NOT NULL, display_label TEXT, resolution_status TEXT NOT NULL, public_player_id TEXT, observation_count INTEGER NOT NULL, observations_json TEXT NOT NULL, PRIMARY KEY (family, game_id))"
                )
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS counts (scope TEXT NOT NULL, family TEXT NOT NULL, public_player_id TEXT NOT NULL, condition TEXT NOT NULL, outcome TEXT NOT NULL, count INTEGER NOT NULL CHECK (count > 0), PRIMARY KEY (scope, family, public_player_id, condition, outcome))"
                )
                for key, value in expected.items():
                    self._connection.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", (key, value))
                initial_state = _sha({"base_manifest_sha256": expected["base_manifest_sha256"], "contract": LIVE_STATISTICAL_OVERLAY_CONTRACT})
                self._connection.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES ('revision', '0')")
                self._connection.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES ('state_sha256', ?)", (initial_state,))
                actual = {str(row["key"]): str(row["value"]) for row in self._connection.execute("SELECT key, value FROM metadata")}
                for key, value in expected.items():
                    if actual.get(key) != value:
                        raise RuntimeError(f"statistical-package overlay metadata differs for {key}: {actual.get(key)!r} != {value!r}")
                self._connection.commit()
            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise

    def _metadata(self, key: str) -> str:
        row = self._connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise RuntimeError(f"statistical-package overlay metadata is missing {key}")
        return str(row["value"])

    def update_completed_game(self, final_game: Mapping[str, object], *, completed_at: str, completion_order: int) -> dict[str, object]:
        """Atomically apply one terminal game exactly once and return its durable receipt."""
        family = str(final_game.get("game_family") or "")
        game_id = str(final_game.get("game_id") or "")
        if family not in GLEE_FAMILIES:
            raise ValueError(f"unsupported statistical-package family: {family}")
        if not game_id:
            raise ValueError("completed statistical-package update requires a game_id")
        if final_game.get("result") is None and str(final_game.get("status") or "") not in {"completed", "no_deal"}:
            raise ValueError("statistical-package update requires a terminal game")
        observations = Counter(observation for move in extract_behavior_moves(final_game) for observation in _action_observations(move))
        serialized_observations = [{"condition": condition, "outcome": outcome, "count": count} for (condition, outcome), count in sorted(observations.items())]
        resolution_status, public_player_id, candidates, display_label = self._resolve(final_game)
        evidence_sha256 = _sha({"family": family, "game_id": game_id, "display_label": display_label, "resolution_status": resolution_status, "public_player_id": public_player_id, "candidate_public_player_ids": candidates, "observations": serialized_observations})
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute("SELECT evidence_sha256 FROM games WHERE family = ? AND game_id = ?", (family, game_id)).fetchone()
                if existing is not None:
                    if str(existing["evidence_sha256"]) != evidence_sha256:
                        raise RuntimeError(f"completed game {family}/{game_id} differs from its applied statistical-package evidence")
                    receipt = self._update_receipt(family=family, game_id=game_id, status="duplicate", resolution_status=resolution_status, public_player_id=public_player_id, candidates=candidates, display_label=display_label, observation_count=sum(observations.values()))
                    self._connection.commit()
                    return receipt
                self._connection.execute(
                    "INSERT INTO games(family, game_id, completed_at, completion_order, evidence_sha256, display_label, resolution_status, public_player_id, observation_count, observations_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (family, game_id, str(completed_at), int(completion_order), evidence_sha256, display_label, resolution_status, public_player_id, sum(observations.values()), _canonical(serialized_observations)),
                )
                for (condition, outcome), count in observations.items():
                    self._increment_count("population", family, "", condition, outcome, count)
                    if public_player_id is not None:
                        self._increment_count("direct", family, public_player_id, condition, outcome, count)
                revision = int(self._metadata("revision")) + 1
                previous_state = self._metadata("state_sha256")
                state_sha256 = _sha({"previous": previous_state, "family": family, "game_id": game_id, "evidence_sha256": evidence_sha256, "public_player_id": public_player_id, "observations": serialized_observations})
                self._connection.execute("UPDATE metadata SET value = ? WHERE key = 'revision'", (str(revision),))
                self._connection.execute("UPDATE metadata SET value = ? WHERE key = 'state_sha256'", (state_sha256,))
                receipt = self._update_receipt(family=family, game_id=game_id, status="updated", resolution_status=resolution_status, public_player_id=public_player_id, candidates=candidates, display_label=display_label, observation_count=sum(observations.values()))
                self._connection.commit()
                return receipt
            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise

    def _increment_count(self, scope: str, family: str, public_player_id: str, condition: str, outcome: str, count: int) -> None:
        self._connection.execute(
            "INSERT INTO counts(scope, family, public_player_id, condition, outcome, count) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(scope, family, public_player_id, condition, outcome) DO UPDATE SET count = count + excluded.count",
            (scope, family, public_player_id, condition, outcome, int(count)),
        )

    def _update_receipt(self, *, family: str, game_id: str, status: str, resolution_status: str, public_player_id: str | None, candidates: list[str], display_label: str | None, observation_count: int) -> dict[str, object]:
        return {
            "contract": LIVE_STATISTICAL_OVERLAY_CONTRACT,
            "status": status,
            "family": family,
            "game_id": game_id,
            "revision": int(self._metadata("revision")),
            "state_sha256": self._metadata("state_sha256"),
            "identity_resolution": {"status": resolution_status, "display_label": display_label, "public_player_id": public_player_id, "candidate_public_player_ids": candidates},
            "observations": observation_count,
            "game_recorded": status == "updated",
            "population_model_updated": status == "updated",
            "population_counts_updated": status == "updated" and observation_count > 0,
            "direct_model_updated": status == "updated" and public_player_id is not None,
            "direct_counts_updated": status == "updated" and observation_count > 0 and public_player_id is not None,
        }

    def _overlay_snapshot(self, family: str, public_player_id: str | None) -> dict[str, object]:
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                revision = int(self._metadata("revision"))
                state_sha256 = self._metadata("state_sha256")
                family_games = int(self._connection.execute("SELECT COUNT(*) AS count FROM games WHERE family = ?", (family,)).fetchone()["count"])
                direct_games = int(self._connection.execute("SELECT COUNT(*) AS count FROM games WHERE family = ? AND public_player_id = ?", (family, public_player_id)).fetchone()["count"]) if public_player_id is not None else 0
                population = self._read_counts("population", family, "")
                direct = self._read_counts("direct", family, public_player_id) if public_player_id is not None else {}
                self._connection.commit()
            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise
        return {"revision": revision, "state_sha256": state_sha256, "family_games": family_games, "direct_games": direct_games, "population_counts": population, "direct_counts": direct}

    def _read_counts(self, scope: str, family: str, public_player_id: str | None) -> dict[str, dict[str, int]]:
        if public_player_id is None:
            return {}
        rows = self._connection.execute("SELECT condition, outcome, count FROM counts WHERE scope = ? AND family = ? AND public_player_id = ? ORDER BY condition, outcome", (scope, family, public_player_id))
        result: dict[str, dict[str, int]] = {}
        for row in rows:
            result.setdefault(str(row["condition"]), {})[str(row["outcome"])] = int(row["count"])
        return result

    def view(self, game: Mapping[str, object]) -> dict[str, object]:
        family = str(game.get("game_family") or "")
        if family not in GLEE_FAMILIES:
            raise ValueError(f"unsupported statistical-package family: {family}")
        resolution, public_player_id, candidates, label = self._resolve(game)
        row = self.rows.get((family, public_player_id)) if public_player_id is not None else None
        overlay = self._overlay_snapshot(family, public_player_id)
        family_population = self.population["families"][family]
        population_contexts = _merged_counts(family_population["contexts"], overlay["population_counts"])
        base_direct = row["direct_counts"] if isinstance(row, Mapping) else {}
        direct_contexts = _merged_counts(base_direct, overlay["direct_counts"])
        visible = _visible_context(game)
        ranked = sorted(population_contexts, key=lambda condition: _context_rank(condition, visible, sum(direct_contexts.get(condition, {}).values()), sum(population_contexts[condition].values())), reverse=True)
        selected = ranked[: self.context_limit if row is not None else min(4, self.context_limit)]
        alpha = float(family_population["alpha"])
        contexts = []
        for condition in selected:
            population_counts = population_contexts[condition]
            direct_counts = direct_contexts.get(condition, {})
            distribution = _smoothed_distribution(direct_counts, population_counts, alpha) if row is not None else _population_distribution(population_counts)
            contexts.append({"condition": condition, "direct_support": sum(int(value) for value in direct_counts.values()), "population_support": sum(int(value) for value in population_counts.values()), "distribution": _project_distribution(distribution, self.outcome_limit)})
        if isinstance(row, Mapping):
            base_evidence = row["evidence"]
            games = int(base_evidence["games"]) + int(overlay["direct_games"])
            observations = int(base_evidence["observations"]) + sum(sum(values.values()) for values in overlay["direct_counts"].values())
            evidence = {"games": games, "observations": observations, "contexts": len(direct_contexts), "tier": _evidence_tier(observations), "base_games": int(base_evidence["games"]), "live_games": int(overlay["direct_games"])}
        else:
            evidence = {"games": 0, "observations": 0, "contexts": 0, "tier": "population-only", "base_games": 0, "live_games": 0}
        result = {
            "contract": self.base_receipt["contract"],
            "release": self.base_receipt["release"],
            "frontier_sequence": self.base_receipt["frontier_sequence"],
            "family": family,
            "identity_resolution": {"status": resolution, "display_label": label, "public_player_id": public_player_id, "candidate_public_player_ids": candidates, "collision_safe": public_player_id is not None or not candidates},
            "evidence": {**evidence, "alpha": alpha},
            "live_overlay": {"contract": LIVE_STATISTICAL_OVERLAY_CONTRACT, "revision": overlay["revision"], "state_sha256": overlay["state_sha256"], "family_completed_games": overlay["family_games"], "direct_opponent_games": overlay["direct_games"]},
            "action_model": {"authority": "advisory", "semantics": "population-smoothed opponent next-action distribution conditional on visible context, including the transactional live overlay", "value_bin_contract": {"index_i_interval": "[-2.0 + 0.1*i, -2.0 + 0.1*(i+1)) with endpoint clipping", "normalized_scale": True}, "visible_context": visible, "contexts": contexts, "projected_contexts": len(contexts), "stored_contexts": len(population_contexts)},
        }
        if family == "bargaining":
            result["decision_local_model"] = _bargaining_decision_model(visible=visible, population_contexts=population_contexts, direct_contexts=direct_contexts, alpha=alpha, identity_exact=row is not None)
        return result

    def status(self) -> dict[str, object]:
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                games = {family: int(self._connection.execute("SELECT COUNT(*) AS count FROM games WHERE family = ?", (family,)).fetchone()["count"]) for family in GLEE_FAMILIES}
                result = {"contract": LIVE_STATISTICAL_OVERLAY_CONTRACT, "revision": int(self._metadata("revision")), "state_sha256": self._metadata("state_sha256"), "games_by_family": games}
                self._connection.commit()
                return result
            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()
