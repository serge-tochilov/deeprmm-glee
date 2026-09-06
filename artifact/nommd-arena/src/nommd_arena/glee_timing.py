"""Compact opponent move-delay evidence and timing-fingerprint inference."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .glee_named_dossier import named_opponent_id, normalize_opponent_name


TIMING_CONTRACT = "glee-opponent-timing-v1"
_EXACT_SOURCE = "server-response-time"
_WALL_SOURCE = "local-causal-wall"
_MAX_ELIGIBLE_WALL_DELAY_MS = 180_000.0


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite_delay(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    delay = float(value)
    return delay if math.isfinite(delay) and delay >= 0.0 else None


def _other_player(player: str) -> str | None:
    return "player_2" if player == "player_1" else "player_1" if player == "player_2" else None


def _opponent_identity(game: dict[str, Any]) -> tuple[str, str | None, str]:
    game_id = str(game.get("game_id") or "")
    opponent = game.get("opponent") if isinstance(game.get("opponent"), dict) else {}
    name = normalize_opponent_name(opponent.get("name"))
    if name and str(opponent.get("type") or "").casefold() != "hidden":
        return named_opponent_id(name), name, "named"
    return f"hidden:{game_id}", None, "hidden"


def _context(game: dict[str, Any], *, move_kind: str, round_number: int | None) -> dict[str, object]:
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    history = state.get("history") if isinstance(state.get("history"), list) else []
    valid_actions = game.get("valid_actions") if isinstance(game.get("valid_actions"), dict) else {}
    fields = valid_actions.get("fields") if isinstance(valid_actions.get("fields"), dict) else {}
    prompt_chars = len(str(game.get("prompt") or ""))
    complexity = len(history) + len(fields) + int(bool(state.get("messages_allowed"))) + int(bool(state.get("complete_information"))) + int(bool(state.get("horizon_known"))) + min(10, prompt_chars // 1000)
    compact = {
        "family": str(game.get("game_family") or "unknown"),
        "phase": str(game.get("phase") or state.get("phase") or "unknown"),
        "move_kind": move_kind,
        "round": round_number,
        "history_length": len(history),
        "prompt_chars": prompt_chars,
        "complexity": complexity,
        "valid_action_type": str(valid_actions.get("type") or "unknown"),
        "messages_allowed": bool(state.get("messages_allowed")),
        "complete_information": bool(state.get("complete_information")),
        "horizon_known": bool(state.get("horizon_known")),
    }
    return {**compact, "fingerprint": _sha(compact)[:20]}


def _exact_opponent_responses(game: dict[str, Any]) -> list[dict[str, object]]:
    family = str(game.get("game_family") or "")
    our_player = str(game.get("your_player") or "")
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    history = state.get("history") if isinstance(state.get("history"), list) else []
    rows: list[dict[str, object]] = []
    for index, entry in enumerate(history, start=1):
        if not isinstance(entry, dict):
            continue
        delay_ms = _finite_delay(entry.get("response_time_ms"))
        if delay_ms is None:
            continue
        round_number = int(entry.get("round") or index)
        actor = ""
        move_kind = "response"
        if family == "bargaining":
            offer = entry.get("offer") if isinstance(entry.get("offer"), dict) else {}
            proposer = str(entry.get("proposer") or offer.get("proposer") or "")
            actor = str(entry.get("decided_by") or _other_player(proposer) or "")
            move_kind = "offer-response"
        elif family == "negotiation":
            actor = str(entry.get("decided_by") or "")
            move_kind = "response-counteroffer" if entry.get("counteroffer") is not None else "offer-response"
        elif family == "persuasion":
            player_1_role = str(state.get("player_1_role") or "")
            actor = "player_1" if player_1_role == "buyer" else "player_2"
            move_kind = "buyer-decision"
        if not actor or actor == our_player:
            continue
        rows.append({"round": round_number, "move_kind": move_kind, "delay_ms": delay_ms, "actor": actor})
    return rows


def _wall_move_kind(game: dict[str, Any], *, terminal: bool) -> str:
    if terminal:
        return "terminal-opponent-move"
    family = str(game.get("game_family") or "")
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    phase = str(game.get("phase") or state.get("phase") or "")
    if family in {"bargaining", "negotiation"}:
        return "opponent-proposal" if phase == "decision" else "opponent-response"
    if family == "persuasion":
        return "seller-signal" if phase == "buyer_decision" else "buyer-decision"
    return "opponent-move"


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summary(values: Iterable[float]) -> dict[str, object]:
    delays = [float(value) for value in values]
    if not delays:
        return {"count": 0}
    logs = [math.log1p(value) for value in delays]
    median_ms = statistics.median(delays)
    return {
        "count": len(delays),
        "median_ms": round(median_ms, 3),
        "p10_ms": round(_quantile(delays, 0.1), 3),
        "p90_ms": round(_quantile(delays, 0.9), 3),
        "iqr_log_ms": round(_quantile(logs, 0.75) - _quantile(logs, 0.25), 6),
        "fast_mass_le_2s": round(sum(value <= 2_000.0 for value in delays) / len(delays), 6),
        "model_scale_mass_ge_8s": round(sum(value >= 8_000.0 for value in delays) / len(delays), 6),
        "long_mass_ge_30s": round(sum(value >= 30_000.0 for value in delays) / len(delays), 6),
    }


def _engine_hint(exact_summary: dict[str, object]) -> str:
    count = int(exact_summary.get("count") or 0)
    if count < 3:
        return "insufficient-evidence"
    fast_mass = float(exact_summary.get("fast_mass_le_2s") or 0.0)
    model_mass = float(exact_summary.get("model_scale_mass_ge_8s") or 0.0)
    p90 = float(exact_summary.get("p90_ms") or 0.0)
    median = float(exact_summary.get("median_ms") or 0.0)
    if fast_mass >= 0.8 and p90 <= 4_000.0:
        return "deterministic-like"
    if model_mass >= 0.6 or median >= 8_000.0:
        return "model-call-like"
    return "mixed-delayed-or-hybrid"


class OpponentTimingStore:
    """Store compact timing observations without copying full game records."""

    def __init__(self, root: Path, *, poll_resolution_s: float = 2.0) -> None:
        if poll_resolution_s < 0:
            raise ValueError("poll_resolution_s cannot be negative")
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "timing.sqlite3"
        self.poll_resolution_ms = poll_resolution_s * 1000.0
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                opponent_id TEXT NOT NULL,
                opponent_name TEXT,
                identity_kind TEXT NOT NULL,
                family TEXT NOT NULL,
                game_id TEXT NOT NULL,
                round_number INTEGER,
                move_kind TEXT NOT NULL,
                timing_source TEXT NOT NULL,
                delay_ms REAL NOT NULL,
                resolution_ms REAL NOT NULL,
                eligible INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                source_run TEXT NOT NULL,
                source_event_sequence INTEGER,
                source_turn_id TEXT,
                phase TEXT NOT NULL,
                history_length INTEGER NOT NULL,
                prompt_chars INTEGER NOT NULL,
                complexity INTEGER NOT NULL,
                context_fingerprint TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS observations_identity_family ON observations(opponent_id, family, timing_source, move_kind, observed_at);
            CREATE INDEX IF NOT EXISTS observations_game ON observations(game_id, observed_at);
            CREATE TABLE IF NOT EXISTS submission_anchors (
                game_id TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                submitted_turn_id TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                source_run TEXT NOT NULL,
                source_event_sequence INTEGER
            );
            """
        )
        self.connection.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES ('contract', ?)", (TIMING_CONTRACT,))
        contract = self.connection.execute("SELECT value FROM metadata WHERE key = 'contract'").fetchone()
        if contract is None or str(contract["value"]) != TIMING_CONTRACT:
            raise RuntimeError(f"opponent timing store contract mismatch: {dict(contract) if contract is not None else None}")
        self.connection.execute("PRAGMA user_version=1")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def _insert(
        self,
        *,
        game: dict[str, Any],
        round_number: int | None,
        move_kind: str,
        timing_source: str,
        delay_ms: float,
        resolution_ms: float,
        eligible: bool,
        observed_at: str,
        source_run: str,
        source_event_sequence: int | None,
        source_turn_id: str | None,
        discriminator: str,
    ) -> bool:
        opponent_id, opponent_name, identity_kind = _opponent_identity(game)
        family = str(game.get("game_family") or "unknown")
        game_id = str(game.get("game_id") or "")
        context = _context(game, move_kind=move_kind, round_number=round_number)
        observation_id = _sha({"contract": TIMING_CONTRACT, "game_id": game_id, "source": timing_source, "discriminator": discriminator})
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO observations (
                observation_id, opponent_id, opponent_name, identity_kind, family, game_id, round_number, move_kind, timing_source, delay_ms, resolution_ms, eligible, observed_at, source_run, source_event_sequence, source_turn_id, phase, history_length, prompt_chars, complexity, context_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                opponent_id,
                opponent_name,
                identity_kind,
                family,
                game_id,
                round_number,
                move_kind,
                timing_source,
                delay_ms,
                resolution_ms,
                int(eligible),
                observed_at,
                source_run,
                source_event_sequence,
                source_turn_id,
                context["phase"],
                context["history_length"],
                context["prompt_chars"],
                context["complexity"],
                context["fingerprint"],
            ),
        )
        return cursor.rowcount == 1

    def record_submission(self, *, game: dict[str, Any], turn_id: str, submitted_at: str, source_run: str, source_event_sequence: int | None) -> None:
        self.connection.execute(
            """
            INSERT INTO submission_anchors(game_id, family, submitted_turn_id, submitted_at, source_run, source_event_sequence) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET family=excluded.family, submitted_turn_id=excluded.submitted_turn_id, submitted_at=excluded.submitted_at, source_run=excluded.source_run, source_event_sequence=excluded.source_event_sequence
            """,
            (str(game["game_id"]), str(game["game_family"]), turn_id, submitted_at, source_run, source_event_sequence),
        )
        self.connection.commit()

    def observe_turn(
        self,
        *,
        game: dict[str, Any],
        turn_id: str,
        observed_at: str,
        source_run: str,
        source_event_sequence: int | None,
        terminal: bool = False,
    ) -> dict[str, object]:
        inserted_exact = 0
        for row in _exact_opponent_responses(game):
            inserted_exact += int(
                self._insert(
                    game=game,
                    round_number=int(row["round"]),
                    move_kind=str(row["move_kind"]),
                    timing_source=_EXACT_SOURCE,
                    delay_ms=float(row["delay_ms"]),
                    resolution_ms=0.0,
                    eligible=True,
                    observed_at=observed_at,
                    source_run=source_run,
                    source_event_sequence=source_event_sequence,
                    source_turn_id=turn_id,
                    discriminator=f"round:{row['round']}:actor:{row['actor']}:kind:{row['move_kind']}",
                )
            )
        wall_inserted = False
        anchor = self.connection.execute("SELECT * FROM submission_anchors WHERE game_id = ?", (str(game["game_id"]),)).fetchone()
        if anchor is not None and str(anchor["submitted_turn_id"]) != turn_id:
            delay_ms = max(0.0, (_parse_time(observed_at) - _parse_time(str(anchor["submitted_at"]))).total_seconds() * 1000.0)
            wall_inserted = self._insert(
                game=game,
                round_number=int((game.get("game_state") or {}).get("round") or 0) or None,
                move_kind=_wall_move_kind(game, terminal=terminal),
                timing_source=_WALL_SOURCE,
                delay_ms=delay_ms,
                resolution_ms=self.poll_resolution_ms,
                eligible=delay_ms <= _MAX_ELIGIBLE_WALL_DELAY_MS,
                observed_at=observed_at,
                source_run=source_run,
                source_event_sequence=source_event_sequence,
                source_turn_id=turn_id,
                discriminator=f"after:{anchor['submitted_turn_id']}:before:{turn_id}",
            )
            self.connection.execute("DELETE FROM submission_anchors WHERE game_id = ?", (str(game["game_id"]),))
        self.connection.commit()
        opponent_id, _name, identity_kind = _opponent_identity(game)
        profile = self.profile(opponent_id=opponent_id, family=str(game.get("game_family") or "unknown"))
        candidates = self.timing_candidates(game_id=str(game["game_id"]), family=str(game.get("game_family") or "unknown")) if identity_kind == "hidden" and (inserted_exact or wall_inserted) else []
        hardness = self.latest_hardness(opponent_id=opponent_id, family=str(game.get("game_family") or "unknown")) if inserted_exact else None
        return {"contract": TIMING_CONTRACT, "inserted_exact": inserted_exact, "inserted_wall": wall_inserted, "profile": profile, "latest_hardness": hardness, "timing_candidates": candidates}

    def profile(self, *, opponent_id: str, family: str | None = None) -> dict[str, object]:
        parameters: list[object] = [opponent_id]
        clause = "opponent_id = ? AND eligible = 1"
        if family is not None:
            clause += " AND family = ?"
            parameters.append(family)
        rows = self.connection.execute(f"SELECT timing_source, move_kind, delay_ms FROM observations WHERE {clause}", parameters).fetchall()
        exact = [float(row["delay_ms"]) for row in rows if row["timing_source"] == _EXACT_SOURCE]
        wall = [float(row["delay_ms"]) for row in rows if row["timing_source"] == _WALL_SOURCE]
        groups: list[dict[str, object]] = []
        for source, move_kind in sorted({(str(row["timing_source"]), str(row["move_kind"])) for row in rows}):
            values = [float(row["delay_ms"]) for row in rows if row["timing_source"] == source and row["move_kind"] == move_kind]
            groups.append({"timing_source": source, "move_kind": move_kind, **_summary(values)})
        value = {
            "contract": TIMING_CONTRACT,
            "opponent_id": opponent_id,
            "family": family,
            "exact": _summary(exact),
            "causal_wall": _summary(wall),
            "groups": groups,
            "engine_hint": _engine_hint(_summary(exact)),
            "interpretation": "Engine and state-hardness labels are timing hypotheses, not substrate facts; exact server response times outrank poll-censored local wall intervals.",
        }
        value["fingerprint_sha256"] = _sha(value)
        return value

    def latest_hardness(self, *, opponent_id: str, family: str) -> dict[str, object] | None:
        latest = self.connection.execute(
            "SELECT observation_id, move_kind, delay_ms, complexity, observed_at FROM observations WHERE opponent_id = ? AND family = ? AND timing_source = ? AND eligible = 1 ORDER BY observed_at DESC, observation_id DESC LIMIT 1",
            (opponent_id, family, _EXACT_SOURCE),
        ).fetchone()
        if latest is None:
            return None
        comparable = self.connection.execute(
            "SELECT delay_ms, complexity FROM observations WHERE opponent_id = ? AND family = ? AND timing_source = ? AND move_kind = ? AND eligible = 1 AND observation_id != ?",
            (opponent_id, family, _EXACT_SOURCE, latest["move_kind"], latest["observation_id"]),
        ).fetchall()
        near = [float(row["delay_ms"]) for row in comparable if abs(int(row["complexity"]) - int(latest["complexity"])) <= 2]
        baseline = near if len(near) >= 3 else [float(row["delay_ms"]) for row in comparable]
        if len(baseline) < 3:
            return {"status": "insufficient-baseline", "move_kind": str(latest["move_kind"]), "delay_ms": round(float(latest["delay_ms"]), 3), "baseline_count": len(baseline)}
        center = statistics.median(math.log1p(value) for value in baseline)
        residual = math.log1p(float(latest["delay_ms"])) - center
        label = "hard-state-like" if residual >= math.log(2.0) else "easy-state-like" if residual <= -math.log(2.0) else "typical-latency"
        engine_hint = str(self.profile(opponent_id=opponent_id, family=family)["engine_hint"])
        return {
            "status": "estimated" if engine_hint != "deterministic-like" else "not-interpreted-for-deterministic-like-profile",
            "move_kind": str(latest["move_kind"]),
            "delay_ms": round(float(latest["delay_ms"]), 3),
            "baseline_count": len(baseline),
            "baseline_scope": "same-kind-near-complexity" if baseline is near else "same-kind",
            "log_latency_residual": round(residual, 6),
            "hardness_hint": label,
            "engine_hint": engine_hint,
        }

    def timing_candidates(self, *, game_id: str, family: str, limit: int = 5) -> list[dict[str, object]]:
        hidden_id = f"hidden:{game_id}"
        hidden = self.connection.execute("SELECT timing_source, move_kind, delay_ms, complexity FROM observations WHERE opponent_id = ? AND family = ? AND eligible = 1", (hidden_id, family)).fetchall()
        if not hidden:
            return []
        all_candidate_rows = self.connection.execute("SELECT opponent_id, timing_source, move_kind, delay_ms, complexity, opponent_name FROM observations WHERE identity_kind = 'named' AND family = ? AND eligible = 1", (family,)).fetchall()
        candidate_groups: dict[str, list[sqlite3.Row]] = {}
        for row in all_candidate_rows:
            candidate_groups.setdefault(str(row["opponent_id"]), []).append(row)
        scored: list[dict[str, object]] = []
        for candidate_id, candidate_rows in candidate_groups.items():
            distances: list[float] = []
            matched = 0
            for source in (_EXACT_SOURCE, _WALL_SOURCE):
                for move_kind in {str(row["move_kind"]) for row in hidden if row["timing_source"] == source}:
                    target = [math.log1p(float(row["delay_ms"])) for row in hidden if row["timing_source"] == source and row["move_kind"] == move_kind]
                    matching_rows = [row for row in candidate_rows if row["timing_source"] == source and row["move_kind"] == move_kind]
                    target_complexity = round(statistics.median(int(row["complexity"]) for row in hidden if row["timing_source"] == source and row["move_kind"] == move_kind))
                    near_rows = [row for row in matching_rows if abs(int(row["complexity"]) - target_complexity) <= 2]
                    reference_rows = near_rows if len(near_rows) >= 2 else matching_rows
                    reference = [math.log1p(float(row["delay_ms"])) for row in reference_rows]
                    if not target or len(reference) < 2:
                        continue
                    center = statistics.median(reference)
                    mad = statistics.median(abs(value - center) for value in reference)
                    scale = max(0.45, 1.4826 * mad)
                    source_weight = 1.0 if source == _EXACT_SOURCE else 0.35
                    distances.extend(source_weight * ((value - center) / scale) ** 2 for value in target)
                    matched += len(target)
            if not distances:
                continue
            mean_distance = sum(distances) / len(distances)
            scored.append(
                {
                    "opponent_id": candidate_id,
                    "opponent_name": next((str(row["opponent_name"]) for row in candidate_rows if row["opponent_name"]), None),
                    "timing_similarity": round(math.exp(-0.5 * mean_distance), 8),
                    "matched_observations": matched,
                    "candidate_observations": len(candidate_rows),
                }
            )
        scored.sort(key=lambda row: (-float(row["timing_similarity"]), -int(row["matched_observations"]), str(row["opponent_id"])))
        return scored[:limit]

    def counts(self) -> dict[str, int]:
        row = self.connection.execute(
            "SELECT COUNT(*) AS observations, COUNT(DISTINCT game_id) AS games, COUNT(DISTINCT CASE WHEN identity_kind = 'named' THEN opponent_id END) AS named_opponents FROM observations"
        ).fetchone()
        return {"observations": int(row["observations"]), "games": int(row["games"]), "named_opponents": int(row["named_opponents"])}

    def ingest_run(self, run_dir: Path) -> dict[str, int]:
        """Extract compact timing rows from one immutable run without altering live submission anchors."""
        events_path = run_dir / "events.jsonl"
        if not events_path.is_file():
            return {"events": 0, "observations": 0, "malformed": 0}
        local_anchors: dict[str, tuple[str, str]] = {}
        latest_games: dict[str, dict[str, Any]] = {}
        events = 0
        inserted = 0
        malformed = 0

        def ingest_game(game: dict[str, Any], *, turn_id: str, observed_at: str, event_sequence: int | None, terminal: bool = False) -> None:
            nonlocal inserted
            game_id = str(game.get("game_id") or "")
            latest_games[game_id] = game
            for row in _exact_opponent_responses(game):
                inserted += int(
                    self._insert(
                        game=game,
                        round_number=int(row["round"]),
                        move_kind=str(row["move_kind"]),
                        timing_source=_EXACT_SOURCE,
                        delay_ms=float(row["delay_ms"]),
                        resolution_ms=0.0,
                        eligible=True,
                        observed_at=observed_at,
                        source_run=str(run_dir),
                        source_event_sequence=event_sequence,
                        source_turn_id=turn_id,
                        discriminator=f"round:{row['round']}:actor:{row['actor']}:kind:{row['move_kind']}",
                    )
                )
            anchor = local_anchors.get(game_id)
            if anchor is not None and anchor[0] != turn_id:
                delay_ms = max(0.0, (_parse_time(observed_at) - _parse_time(anchor[1])).total_seconds() * 1000.0)
                inserted += int(
                    self._insert(
                        game=game,
                        round_number=int((game.get("game_state") or {}).get("round") or 0) or None,
                        move_kind=_wall_move_kind(game, terminal=terminal),
                        timing_source=_WALL_SOURCE,
                        delay_ms=delay_ms,
                        resolution_ms=self.poll_resolution_ms,
                        eligible=delay_ms <= _MAX_ELIGIBLE_WALL_DELAY_MS,
                        observed_at=observed_at,
                        source_run=str(run_dir),
                        source_event_sequence=event_sequence,
                        source_turn_id=turn_id,
                        discriminator=f"after:{anchor[0]}:before:{turn_id}",
                    )
                )
                del local_anchors[game_id]

        with events_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(event, dict):
                    malformed += 1
                    continue
                events += 1
                kind = str(event.get("kind") or "")
                event_sequence = int(event["event_sequence"]) if isinstance(event.get("event_sequence"), int) else None
                observed_at = str(event.get("ts") or "")
                if kind == "turn_observed" and isinstance(event.get("game"), dict) and observed_at:
                    ingest_game(event["game"], turn_id=str(event.get("turn_id") or "unknown-turn"), observed_at=observed_at, event_sequence=event_sequence)
                    continue
                if kind == "move_submitted" and isinstance(event.get("result"), dict):
                    result = event["result"]
                    game_id = str(event.get("game_id") or "")
                    if result.get("valid") is not False and not result.get("game_over") and game_id in latest_games and observed_at:
                        local_anchors[game_id] = (str(event.get("turn_id") or "unknown-turn"), observed_at)
                    continue
                if kind == "server_rejection_fallback" and isinstance(event.get("result"), dict):
                    result = event["result"]
                    turn_id = str(event.get("turn_id") or "")
                    game_id = turn_id.partition(":")[0]
                    if result.get("valid") is not False and not result.get("game_over") and game_id in latest_games and observed_at:
                        local_anchors[game_id] = (turn_id or "unknown-turn", observed_at)
                    continue
                if kind in {"game_completed", "game_completed_during_opponent_turn", "late_game_reconciled"}:
                    game_id = str(event.get("game_id") or "")
                    family = str(event.get("family") or "unknown")
                    game_path = run_dir / "games" / f"{family}-{game_id}.json"
                    if game_path.is_file() and observed_at:
                        try:
                            final_game = json.loads(game_path.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, OSError):
                            malformed += 1
                        else:
                            if isinstance(final_game, dict):
                                ingest_game(final_game, turn_id=f"{game_id}:terminal", observed_at=observed_at, event_sequence=event_sequence, terminal=kind != "game_completed")
        self.connection.commit()
        return {"events": events, "observations": inserted, "malformed": malformed}


def bootstrap_timing_store(*, source_root: Path, output_root: Path, poll_resolution_s: float = 4.0) -> dict[str, object]:
    """Populate one deduplicated timing store from historical run references."""
    store = OpponentTimingStore(output_root, poll_resolution_s=poll_resolution_s)
    totals = {"runs": 0, "events": 0, "observations_inserted": 0, "malformed": 0}
    try:
        event_paths = sorted(source_root.glob("*/events.jsonl"), key=lambda path: (path.stat().st_mtime_ns, str(path)))
        for events_path in event_paths:
            receipt = store.ingest_run(events_path.parent)
            if receipt["events"] == 0:
                continue
            totals["runs"] += 1
            totals["events"] += receipt["events"]
            totals["observations_inserted"] += receipt["observations"]
            totals["malformed"] += receipt["malformed"]
        return {"contract": TIMING_CONTRACT, "source_root": str(source_root), "output_root": str(output_root), **totals, "store": store.counts()}
    finally:
        store.close()
