"""Atomic read-only GLEE sensor frontier shared by independent family services."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from glee_sdk import GleeClient

from .glee import load_glee_api_key
from .glee_api_limiter import AgentWideGleeAPIGateway, AgentWideGleeAPIRateLimiter


SCHEMA_VERSION = 1
SENSOR_KIND = "glee-sensor-frontier"
SENSOR_CONTRACT = "glee-sensor-frontier-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class GleeSensorReader:
    """Read one versioned atomic frontier without owning a GLEE credential."""

    def __init__(self, root: Path, *, max_age_s: float = 8.0) -> None:
        if max_age_s <= 0:
            raise ValueError("sensor max age must be positive")
        self.root = root
        self.current_path = root / "current.json"
        self.max_age_s = max_age_s

    def read(self) -> dict[str, Any] | None:
        if not self.current_path.is_file():
            return None
        value = json.loads(self.current_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != SENSOR_KIND or value.get("contract") != SENSOR_CONTRACT:
            raise RuntimeError(f"unsupported GLEE sensor frontier: {self.current_path}")
        expected = value.get("frontier_sha256")
        actual = _digest({key: item for key, item in value.items() if key != "frontier_sha256"})
        if expected != actual:
            raise RuntimeError(f"GLEE sensor frontier digest mismatch: {self.current_path}")
        fetched_at = datetime.fromisoformat(str(value["fetched_at"]))
        age_s = max(0.0, (datetime.now(timezone.utc) - fetched_at).total_seconds())
        if age_s > self.max_age_s:
            return None
        pending = value.get("pending_games")
        stats = value.get("stats")
        if not isinstance(pending, list) or not isinstance(stats, dict):
            raise RuntimeError(f"malformed GLEE sensor frontier: {self.current_path}")
        return value


class GleeSensorFeed:
    """Own normal pending-game and statistics polling for all family services."""

    def __init__(
        self,
        *,
        project_root: Path,
        output_root: Path,
        env_file: Path | None,
        poll_interval_s: float = 2.0,
        stats_interval_s: float = 15.0,
        api_rate_limit_state: Path | None = None,
        api_request_limit: int = 60,
        api_window_s: float = 60.0,
        api_move_reserve: int = 8,
        api_control_reserve: int = 8,
        client: Any | None = None,
    ) -> None:
        if poll_interval_s <= 0 or stats_interval_s <= 0:
            raise ValueError("sensor polling intervals must be positive")
        self.project_root = project_root
        self.output_root = output_root
        self.poll_interval_s = poll_interval_s
        self.stats_interval_s = stats_interval_s
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.current_path = self.output_root / "current.json"
        self.events_path = self.output_root / "events.jsonl"
        self.manifest_path = self.output_root / "manifest.json"
        self.stop_request_path = self.output_root / "stop.requested"
        self.complete_path = self.output_root / "complete.json"
        self._event_sequence = sum(1 for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()) if self.events_path.is_file() else 0
        if client is None:
            base_url = os.environ.get("GLEE_API_URL")
            client_args: dict[str, Any] = {"api_key": load_glee_api_key(project_root, env_file), "timeout": 10}
            if base_url:
                client_args["base_url"] = base_url
            client = GleeClient(**client_args)
        self.client = client
        self.api_rate_limit_state = api_rate_limit_state.resolve() if api_rate_limit_state is not None else None
        self.api_rate_limiter = AgentWideGleeAPIRateLimiter(state_path=self.api_rate_limit_state, request_limit=api_request_limit, window_s=api_window_s, move_reserve=api_move_reserve, control_reserve=api_control_reserve) if self.api_rate_limit_state is not None else None
        self.api_gateway = AgentWideGleeAPIGateway(client=self.client, limiter=self.api_rate_limiter) if self.api_rate_limiter is not None else None
        os.environ.pop("GLEE_API_KEY", None)
        try:
            self._write_or_validate_manifest()
        except BaseException:
            self.close()
            raise

    def _write_or_validate_manifest(self) -> None:
        configuration = {
            "schema_version": SCHEMA_VERSION,
            "kind": "glee-sensor-manifest",
            "contract": SENSOR_CONTRACT,
            "poll_interval_s": self.poll_interval_s,
            "stats_interval_s": self.stats_interval_s,
            "agent_wide_api_rate_limiter": self.api_gateway.manifest_receipt() if self.api_gateway is not None else None,
        }
        if self.manifest_path.is_file():
            actual = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            comparable = {key: actual.get(key) for key in configuration}
            if comparable != configuration:
                raise RuntimeError(f"GLEE sensor configuration differs on resume: {comparable!r} != {configuration!r}")
            return
        _atomic_json(self.manifest_path, {**configuration, "started_at": _now()})

    def _event(self, kind: str, **values: object) -> dict[str, object]:
        self._event_sequence += 1
        record = {"schema_version": SCHEMA_VERSION, "event_sequence": self._event_sequence, "ts": _now(), "kind": kind, **values}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def _api_call(self, operation: str, callback: Any) -> Any:
        if self.api_gateway is None:
            return callback()
        return self.api_gateway.call(operation=operation, priority="control", callback=callback)

    def run(self, *, max_cycles: int | None = None) -> dict[str, object]:
        if max_cycles is not None and max_cycles < 1:
            raise ValueError("sensor max cycles must be positive")
        self.complete_path.unlink(missing_ok=True)
        sequence = 0
        cached_stats: dict[str, Any] | None = None
        last_stats_at = float("-inf")
        self._event("sensor_started", pid=os.getpid())
        while not self.stop_request_path.is_file():
            cycle_started = time.monotonic()
            try:
                pending = self._api_call("sensor_pending_games", self.client.pending_games)
                if cached_stats is None or cycle_started - last_stats_at >= self.stats_interval_s:
                    cached_stats = self._api_call("sensor_stats", self.client.stats)
                    last_stats_at = cycle_started
                    self._event("agent_stats_observed", stats=cached_stats, stats_sha256=_digest(cached_stats))
                sequence += 1
                frontier: dict[str, object] = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": SENSOR_KIND,
                    "contract": SENSOR_CONTRACT,
                    "sequence": sequence,
                    "fetched_at": _now(),
                    "producer_pid": os.getpid(),
                    "pending_games": pending,
                    "stats": cached_stats,
                }
                frontier["frontier_sha256"] = _digest(frontier)
                _atomic_json(self.current_path, frontier)
                self._event("frontier_published", sequence=sequence, pending_games=len(pending), active_games=cached_stats.get("active_games"))
            except Exception as error:
                self._event("sensor_poll_failed", error=f"{type(error).__name__}: {error}")
            if max_cycles is not None and sequence >= max_cycles:
                break
            delay = max(0.0, self.poll_interval_s - (time.monotonic() - cycle_started))
            if delay:
                time.sleep(delay)
        result = {"schema_version": SCHEMA_VERSION, "kind": "glee-sensor-complete", "contract": SENSOR_CONTRACT, "completed_at": _now(), "frontier_sequence": sequence, "stop_requested": self.stop_request_path.is_file()}
        _atomic_json(self.complete_path, result)
        self._event("sensor_stopped", frontier_sequence=sequence, stop_requested=result["stop_requested"])
        return result

    def close(self) -> None:
        if self.api_gateway is not None:
            self.api_gateway.close()
        if self.api_rate_limiter is not None:
            self.api_rate_limiter.close()
