"""Restart-stable independent Poisson clocks for concealed GLEE admissions."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
from pathlib import Path
from typing import Callable, Mapping


ACTIVITY_SCHEDULER_CONTRACT = "glee-independent-poisson-activity-scheduler-v1"
LIVE_ACTIVITY_TARGET_CONTROL_CONTRACT = "glee-live-activity-target-control-v1"
DEFAULT_WINDOW_S = 48.0 * 60.0 * 60.0
DEFAULT_MINIMUM_TARGET = 100


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class IndependentActivityScheduler:
    """Generate independent Poisson arrivals and retain them until dispatch."""

    def __init__(
        self,
        *,
        state_path: Path,
        targets: Mapping[str, int],
        window_s: float = DEFAULT_WINDOW_S,
        minimum_target: int = DEFAULT_MINIMUM_TARGET,
        clock: Callable[[], float],
        seed_hex: str | None = None,
    ) -> None:
        normalized = {str(family): int(target) for family, target in targets.items()}
        if not normalized:
            raise ValueError("activity scheduler requires at least one family target")
        if isinstance(minimum_target, bool) or minimum_target < 1:
            raise ValueError("activity minimum target must be a positive integer")
        if not math.isfinite(window_s) or window_s <= 0.0:
            raise ValueError("activity window must be positive and finite")
        if any(isinstance(targets[family], bool) or target < minimum_target for family, target in normalized.items()):
            raise ValueError(f"every activity target must be an integer of at least {minimum_target}")
        self.state_path = state_path
        self.configured_targets = normalized
        self.targets = dict(normalized)
        self.window_s = float(window_s)
        self.minimum_target = int(minimum_target)
        self.clock = clock
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self._validate_state(state)
            self._state = state
            self.targets = {str(family): int(target) for family, target in state["targets"].items()}
        else:
            seed = seed_hex or secrets.token_hex(32)
            if len(seed) < 32:
                raise ValueError("activity scheduler seed must contain at least 128 bits")
            int(seed, 16)
            now = self.clock()
            families: dict[str, object] = {}
            for family in sorted(self.targets):
                interval = self._sample_interval(seed, family, 0)
                families[family] = {
                    "next_event_index": 0,
                    "next_due_at_unix": now + interval,
                    "next_interval_s": interval,
                    "pending": [],
                    "outstanding": None,
                    "arrivals": 0,
                    "dispatched": 0,
                    "matched": 0,
                    "queue_errors": 0,
                    "suppressed_on_pause": 0,
                    "offline_rebases": 0,
                    "outstanding_recoveries": 0,
                    "target_changes": 0,
                }
            self._state = {
                "schema_version": 1,
                "contract": ACTIVITY_SCHEDULER_CONTRACT,
                "distribution": "exponential-interarrival-v1",
                "window_s": self.window_s,
                "minimum_target": self.minimum_target,
                "baseline_targets": self.configured_targets,
                "targets": self.targets,
                "seed_hex": seed,
                "paused": False,
                "families": families,
            }
            self._write()

    def _validate_state(self, state: object) -> None:
        if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("contract") != ACTIVITY_SCHEDULER_CONTRACT:
            raise RuntimeError("activity scheduler state has an incompatible contract")
        state_targets = state.get("targets")
        baseline_targets = state.get("baseline_targets", state_targets)
        if baseline_targets != self.configured_targets or float(state.get("window_s") or 0.0) != self.window_s or int(state.get("minimum_target") or 0) != self.minimum_target:
            raise RuntimeError("activity scheduler resume configuration differs")
        seed = state.get("seed_hex")
        families = state.get("families")
        valid_targets = isinstance(state_targets, dict) and set(state_targets) == set(self.configured_targets) and all(isinstance(target, int) and not isinstance(target, bool) and target >= self.minimum_target for target in state_targets.values())
        if not valid_targets or not isinstance(seed, str) or len(seed) < 32 or not isinstance(families, dict) or set(families) != set(self.configured_targets):
            raise RuntimeError("activity scheduler state is incomplete")
        int(seed, 16)
        for family, record in families.items():
            if not isinstance(record, dict) or not isinstance(record.get("next_event_index"), int) or not isinstance(record.get("next_due_at_unix"), (int, float)) or not isinstance(record.get("pending"), list) or (record.get("outstanding") is not None and not isinstance(record.get("outstanding"), dict)):
                raise RuntimeError(f"activity scheduler state for {family} is invalid")
            for arrival in record["pending"]:
                if not isinstance(arrival, dict) or not isinstance(arrival.get("event_index"), int) or not isinstance(arrival.get("scheduled_at_unix"), (int, float)):
                    raise RuntimeError(f"pending activity arrival for {family} is invalid")

    @staticmethod
    def _uniform(seed_hex: str, family: str, event_index: int) -> float:
        digest = hmac.new(bytes.fromhex(seed_hex), f"{family}:{event_index}".encode("utf-8"), hashlib.sha256).digest()
        integer = int.from_bytes(digest[:8], "big")
        return (integer + 0.5) / float(1 << 64)

    def _sample_interval(self, seed_hex: str, family: str, event_index: int, *, target: int | None = None, stream: str | None = None) -> float:
        mean = self.window_s / (self.targets[family] if target is None else target)
        return -mean * math.log1p(-self._uniform(seed_hex, stream or family, event_index))

    def _write(self) -> None:
        _atomic_json(self.state_path, self._state)

    def next_due_at(self, family: str) -> float:
        if family not in self.targets:
            raise KeyError(family)
        return float(self._state["families"][family]["next_due_at_unix"])

    def seconds_until_next_arrival(self, *, now: float | None = None) -> float | None:
        if bool(self._state.get("paused")):
            return None
        instant = self.clock() if now is None else float(now)
        return max(0.0, min(self.next_due_at(family) for family in self.targets) - instant)

    def materialize_due(self, family: str, *, now: float | None = None) -> list[dict[str, object]]:
        if family not in self.targets:
            raise KeyError(family)
        if bool(self._state.get("paused")):
            return []
        instant = self.clock() if now is None else float(now)
        record = self._state["families"][family]
        arrivals: list[dict[str, object]] = []
        seed = str(self._state["seed_hex"])
        while float(record["next_due_at_unix"]) <= instant:
            event_index = int(record["next_event_index"])
            scheduled_at = float(record["next_due_at_unix"])
            arrival = {"event_index": event_index, "scheduled_at_unix": scheduled_at, "target_per_window": self.targets[family]}
            record["pending"].append(arrival)
            arrivals.append({"contract": ACTIVITY_SCHEDULER_CONTRACT, "family": family, **arrival, "observed_at_unix": instant, "observation_lateness_s": round(max(0.0, instant - scheduled_at), 6)})
            record["arrivals"] = int(record.get("arrivals") or 0) + 1
            next_event_index = event_index + 1
            interval = self._sample_interval(seed, family, next_event_index)
            record["next_event_index"] = next_event_index
            record["next_interval_s"] = interval
            record["next_due_at_unix"] = scheduled_at + interval
            if len(arrivals) > 100_000:
                raise RuntimeError("activity scheduler accumulated an implausibly large due-arrival batch")
        if arrivals:
            self._write()
        return arrivals

    def pending_count(self, family: str) -> int:
        if family not in self.targets:
            raise KeyError(family)
        return len(self._state["families"][family]["pending"])

    def pending_total(self) -> int:
        """Return all retained arrivals across independent family clocks."""
        return sum(self.pending_count(family) for family in self.targets)

    def oldest_pending_family(self, *, excluded: set[str] | frozenset[str] = frozenset()) -> str | None:
        """Return the dispatchable family whose oldest retained arrival is globally earliest."""
        candidates = []
        for family in self.targets:
            record = self._state["families"][family]
            if family in excluded or record.get("outstanding") is not None or not record["pending"]:
                continue
            arrival = record["pending"][0]
            candidates.append((float(arrival["scheduled_at_unix"]), int(arrival["event_index"]), family))
        return min(candidates)[2] if candidates else None

    def dispatch_oldest_global(self, *, excluded: set[str] | frozenset[str] = frozenset(), now: float | None = None) -> dict[str, object] | None:
        """Dispatch the globally oldest eligible retained arrival without reserving family capacity."""
        family = self.oldest_pending_family(excluded=excluded)
        if family is None:
            return None
        return {**self.dispatch_oldest(family, now=now), "allocation": "unified-global-fifo"}

    def dispatch_oldest(self, family: str, *, now: float | None = None) -> dict[str, object]:
        if family not in self.targets:
            raise KeyError(family)
        instant = self.clock() if now is None else float(now)
        record = self._state["families"][family]
        pending = record["pending"]
        if not pending:
            raise RuntimeError(f"no pending activity arrival for {family}")
        if record.get("outstanding") is not None:
            raise RuntimeError(f"an activity admission is already outstanding for {family}")
        arrival = pending.pop(0)
        record["outstanding"] = {**arrival, "dispatched_at_unix": instant}
        record["dispatched"] = int(record.get("dispatched") or 0) + 1
        self._write()
        scheduled_at = float(arrival["scheduled_at_unix"])
        return {
            "contract": ACTIVITY_SCHEDULER_CONTRACT,
            "family": family,
            "event_index": int(arrival["event_index"]),
            "scheduled_at_unix": scheduled_at,
            "dispatched_at_unix": instant,
            "dispatch_lateness_s": round(max(0.0, instant - scheduled_at), 6),
            "pending_after_dispatch": len(pending),
            "target_per_window": int(arrival.get("target_per_window") or self.targets[family]),
            "current_target_per_window": self.targets[family],
            "window_s": self.window_s,
        }

    def has_outstanding(self, family: str) -> bool:
        if family not in self.targets:
            raise KeyError(family)
        return self._state["families"][family].get("outstanding") is not None

    def mark_matched(self, family: str, *, game_id: str, now: float | None = None) -> dict[str, object] | None:
        if family not in self.targets:
            raise KeyError(family)
        instant = self.clock() if now is None else float(now)
        record = self._state["families"][family]
        outstanding = record.get("outstanding")
        if not isinstance(outstanding, dict):
            return None
        record["outstanding"] = None
        record["matched"] = int(record.get("matched") or 0) + 1
        self._write()
        scheduled_at = float(outstanding["scheduled_at_unix"])
        dispatched_at = float(outstanding["dispatched_at_unix"])
        return {
            "contract": ACTIVITY_SCHEDULER_CONTRACT,
            "family": family,
            "game_id": game_id,
            "event_index": int(outstanding["event_index"]),
            "scheduled_at_unix": scheduled_at,
            "dispatched_at_unix": dispatched_at,
            "matched_at_unix": instant,
            "dispatch_lateness_s": round(max(0.0, dispatched_at - scheduled_at), 6),
            "matchmaking_latency_s": round(max(0.0, instant - dispatched_at), 6),
            "start_lateness_s": round(max(0.0, instant - scheduled_at), 6),
        }

    def recover_outstanding_after_restart(self) -> list[dict[str, object]]:
        receipts: list[dict[str, object]] = []
        for family in self.targets:
            record = self._state["families"][family]
            outstanding = record.get("outstanding")
            if not isinstance(outstanding, dict):
                continue
            restored = {"event_index": int(outstanding["event_index"]), "scheduled_at_unix": float(outstanding["scheduled_at_unix"])}
            record["pending"].insert(0, restored)
            record["outstanding"] = None
            record["outstanding_recoveries"] = int(record.get("outstanding_recoveries") or 0) + 1
            receipts.append({"contract": ACTIVITY_SCHEDULER_CONTRACT, "family": family, **restored, "prior_dispatched_at_unix": float(outstanding["dispatched_at_unix"])})
        if receipts:
            self._write()
        return receipts

    def record_queue_error(self, family: str) -> None:
        if family not in self.targets:
            raise KeyError(family)
        record = self._state["families"][family]
        record["queue_errors"] = int(record.get("queue_errors") or 0) + 1
        self._write()

    def return_outstanding(self, family: str, *, queue_error: bool) -> dict[str, object] | None:
        if family not in self.targets:
            raise KeyError(family)
        record = self._state["families"][family]
        outstanding = record.get("outstanding")
        if not isinstance(outstanding, dict):
            return None
        restored = {"event_index": int(outstanding["event_index"]), "scheduled_at_unix": float(outstanding["scheduled_at_unix"])}
        record["pending"].insert(0, restored)
        record["outstanding"] = None
        if queue_error:
            record["queue_errors"] = int(record.get("queue_errors") or 0) + 1
        self._write()
        return {"contract": ACTIVITY_SCHEDULER_CONTRACT, "family": family, **restored, "prior_dispatched_at_unix": float(outstanding["dispatched_at_unix"]), "queue_error": queue_error}

    def set_paused(self, paused: bool, *, now: float | None = None) -> dict[str, object] | None:
        paused = bool(paused)
        if bool(self._state.get("paused")) == paused:
            return None
        instant = self.clock() if now is None else float(now)
        suppressed: dict[str, int] = {}
        if paused:
            for family in self.targets:
                record = self._state["families"][family]
                count = len(record["pending"]) + int(record.get("outstanding") is not None)
                suppressed[family] = count
                record["pending"] = []
                record["outstanding"] = None
                record["suppressed_on_pause"] = int(record.get("suppressed_on_pause") or 0) + count
        else:
            seed = str(self._state["seed_hex"])
            for family in self.targets:
                record = self._state["families"][family]
                event_index = int(record["next_event_index"])
                interval = self._sample_interval(seed, family, event_index)
                record["next_interval_s"] = interval
                record["next_due_at_unix"] = instant + interval
        self._state["paused"] = paused
        self._write()
        return {"contract": ACTIVITY_SCHEDULER_CONTRACT, "paused": paused, "at_unix": instant, "suppressed_pending": suppressed}

    def set_target(self, family: str, target: int, *, request_id: str, now: float | None = None) -> dict[str, object]:
        if family not in self.targets:
            raise KeyError(family)
        if isinstance(target, bool) or not isinstance(target, int) or target < self.minimum_target:
            raise ValueError(f"activity target must be an integer of at least {self.minimum_target}")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("activity target change requires a request id")
        instant = self.clock() if now is None else float(now)
        prior_target = self.targets[family]
        record = self._state["families"][family]
        prior_due_at = float(record["next_due_at_unix"])
        if target == prior_target:
            return {
                "contract": LIVE_ACTIVITY_TARGET_CONTROL_CONTRACT,
                "request_id": request_id,
                "family": family,
                "status": "unchanged",
                "target_per_window": target,
                "window_s": self.window_s,
                "at_unix": instant,
                "next_due_at_unix": prior_due_at,
                "pending_preserved": len(record["pending"]),
                "outstanding_preserved": record.get("outstanding") is not None,
            }
        change_index = int(record.get("target_changes") or 0) + 1
        self.targets[family] = target
        self._state["targets"][family] = target
        record["target_changes"] = change_index
        paused = bool(self._state.get("paused"))
        interval: float | None = None
        if not paused:
            event_index = int(record["next_event_index"])
            interval = self._sample_interval(str(self._state["seed_hex"]), family, event_index, target=target, stream=f"{family}:target-change:{change_index}")
            record["next_interval_s"] = interval
            record["next_due_at_unix"] = instant + interval
        receipt = {
            "contract": LIVE_ACTIVITY_TARGET_CONTROL_CONTRACT,
            "request_id": request_id,
            "family": family,
            "status": "changed",
            "prior_target_per_window": prior_target,
            "target_per_window": target,
            "window_s": self.window_s,
            "at_unix": instant,
            "paused": paused,
            "prior_due_at_unix": prior_due_at,
            "prior_due_was_overdue": prior_due_at <= instant,
            "next_due_at_unix": float(record["next_due_at_unix"]) if not paused else None,
            "next_interval_s": interval,
            "pending_preserved": len(record["pending"]),
            "outstanding_preserved": record.get("outstanding") is not None,
            "target_change_index": change_index,
        }
        record["last_target_change"] = receipt
        self._write()
        return receipt

    def rebase_overdue_after_restart(self, *, now: float | None = None) -> list[dict[str, object]]:
        if bool(self._state.get("paused")):
            return []
        instant = self.clock() if now is None else float(now)
        seed = str(self._state["seed_hex"])
        receipts: list[dict[str, object]] = []
        for family in self.targets:
            record = self._state["families"][family]
            due = float(record["next_due_at_unix"])
            if due > instant:
                continue
            event_index = int(record["next_event_index"])
            interval = self._sample_interval(seed, family, event_index)
            record["next_interval_s"] = interval
            record["next_due_at_unix"] = instant + interval
            record["offline_rebases"] = int(record.get("offline_rebases") or 0) + 1
            receipts.append({"contract": ACTIVITY_SCHEDULER_CONTRACT, "family": family, "prior_due_at_unix": due, "restarted_at_unix": instant, "new_due_at_unix": instant + interval, "event_index": event_index})
        if receipts:
            self._write()
        return receipts

    def manifest_receipt(self) -> dict[str, object]:
        return {
            "contract": ACTIVITY_SCHEDULER_CONTRACT,
            "distribution": "exponential-interarrival-v1",
            "targets": dict(self.configured_targets),
            "window_s": self.window_s,
            "minimum_target": self.minimum_target,
            "mean_interval_s": {family: round(self.window_s / target, 6) for family, target in sorted(self.configured_targets.items())},
            "capacity_policy": "retain-arrival-until-any-family-slot-is-available",
            "live_target_control": LIVE_ACTIVITY_TARGET_CONTROL_CONTRACT,
            "seed_sha256": _sha256_text(str(self._state["seed_hex"])),
            "state_path": str(self.state_path),
        }

    def status(self) -> dict[str, object]:
        return {
            **self.manifest_receipt(),
            "active_targets": dict(self.targets),
            "active_mean_interval_s": {family: round(self.window_s / target, 6) for family, target in sorted(self.targets.items())},
            "paused": bool(self._state.get("paused")),
            "families": {
                family: {
                    **{key: value for key, value in self._state["families"][family].items() if key not in {"pending", "outstanding"}},
                    "pending": len(self._state["families"][family]["pending"]),
                    "outstanding": self._state["families"][family]["outstanding"] is not None,
                    "oldest_pending_at_unix": self._state["families"][family]["pending"][0]["scheduled_at_unix"] if self._state["families"][family]["pending"] else None,
                }
                for family in sorted(self.targets)
            },
        }
