"""Crash-safe agent-wide budgeting for authenticated GLEE API requests."""

from __future__ import annotations

import contextvars
import email.utils
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any, Callable, Literal


API_RATE_LIMITER_CONTRACT = "glee-agent-wide-api-rate-limiter-v1"
APIPriority = Literal["background", "control", "critical"]


class GleeAPIBudgetDeferred(RuntimeError):
    """Signal that a request was withheld locally before reaching GLEE."""

    def __init__(self, receipt: dict[str, object]) -> None:
        self.receipt = receipt
        super().__init__(f"agent-wide API budget deferred {receipt.get('operation')}: {receipt.get('reason')}")


class GleeAPIBudgetBypass(RuntimeError):
    """Reject an authenticated HTTP request that lacks an explicit budget context."""


def _parse_retry_after(value: object, *, now: float) -> float:
    if isinstance(value, str):
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, parsed.timestamp() - now)
    return 2.0


class AgentWideGleeAPIRateLimiter:
    """Enforce one strict sliding-window request ledger across processes."""

    def __init__(
        self,
        *,
        state_path: Path,
        request_limit: int = 60,
        window_s: float = 60.0,
        move_reserve: int = 8,
        control_reserve: int = 8,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(request_limit, bool) or not isinstance(request_limit, int) or request_limit < 3:
            raise ValueError("API request limit must be an integer of at least 3")
        if not math.isfinite(window_s) or window_s <= 0.0:
            raise ValueError("API rate-limit window must be positive and finite")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (move_reserve, control_reserve)):
            raise ValueError("API priority reserves must be nonnegative integers")
        if move_reserve + control_reserve >= request_limit:
            raise ValueError("API priority reserves must leave background capacity")
        self.state_path = state_path.resolve()
        self.request_limit = request_limit
        self.window_s = float(window_s)
        self.move_reserve = move_reserve
        self.control_reserve = control_reserve
        self.ceilings = {"background": request_limit - move_reserve - control_reserve, "control": request_limit - move_reserve, "critical": request_limit}
        self.clock = clock
        self.monotonic = monotonic
        self.sleeper = sleeper
        self._lock = threading.RLock()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.state_path, timeout=5.0, isolation_level=None, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        try:
            self._initialize()
        except BaseException:
            self._connection.close()
            raise
        os.chmod(self.state_path, 0o600)

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute("CREATE TABLE IF NOT EXISTS configuration (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), contract TEXT NOT NULL, request_limit INTEGER NOT NULL, window_s REAL NOT NULL, move_reserve INTEGER NOT NULL, control_reserve INTEGER NOT NULL)")
                self._connection.execute("CREATE TABLE IF NOT EXISTS grants (sequence INTEGER PRIMARY KEY AUTOINCREMENT, granted_at_unix REAL NOT NULL, operation TEXT NOT NULL, priority TEXT NOT NULL CHECK (priority IN ('background', 'control', 'critical')), process_id INTEGER NOT NULL)")
                self._connection.execute("CREATE INDEX IF NOT EXISTS grants_time_idx ON grants(granted_at_unix, sequence)")
                self._connection.execute("CREATE TABLE IF NOT EXISTS server_rate_limits (sequence INTEGER PRIMARY KEY AUTOINCREMENT, observed_at_unix REAL NOT NULL, operation TEXT NOT NULL, retry_after_s REAL NOT NULL, blocked_until_unix REAL NOT NULL)")
                self._connection.execute("CREATE TABLE IF NOT EXISTS runtime_state (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), blocked_until_unix REAL NOT NULL)")
                expected = (API_RATE_LIMITER_CONTRACT, self.request_limit, self.window_s, self.move_reserve, self.control_reserve)
                stored = self._connection.execute("SELECT contract, request_limit, window_s, move_reserve, control_reserve FROM configuration WHERE singleton = 1").fetchone()
                if stored is None:
                    self._connection.execute("INSERT INTO configuration VALUES (1, ?, ?, ?, ?, ?)", expected)
                    self._connection.execute("INSERT INTO runtime_state VALUES (1, 0.0)")
                elif tuple(stored) != expected:
                    raise RuntimeError(f"agent-wide API budget resume configuration differs: {tuple(stored)!r} != {expected!r}")
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def manifest_receipt(self) -> dict[str, object]:
        return {
            "contract": API_RATE_LIMITER_CONTRACT,
            "state_path": str(self.state_path),
            "request_limit": self.request_limit,
            "window_s": self.window_s,
            "move_reserve": self.move_reserve,
            "control_reserve": self.control_reserve,
            "priority_ceilings": dict(self.ceilings),
            "accounting": "strict rolling-window ledger; every SDK HTTP retry consumes a separate grant",
            "failure_policy": "background and control calls defer locally; critical moves may wait only inside their original deadline",
        }

    def _attempt(self, *, operation: str, priority: APIPriority) -> dict[str, object]:
        if not operation:
            raise ValueError("API budget operation must be nonempty")
        if priority not in self.ceilings:
            raise ValueError(f"unsupported API budget priority: {priority}")
        raw_now = float(self.clock())
        if not math.isfinite(raw_now):
            raise ValueError("API budget clock returned a non-finite time")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                latest = self._connection.execute("SELECT MAX(granted_at_unix) FROM grants").fetchone()[0]
                now = max(raw_now, float(latest)) if latest is not None else raw_now
                cutoff = now - self.window_s
                self._connection.execute("DELETE FROM grants WHERE granted_at_unix <= ?", (cutoff,))
                blocked_until = float(self._connection.execute("SELECT blocked_until_unix FROM runtime_state WHERE singleton = 1").fetchone()[0])
                count = int(self._connection.execute("SELECT COUNT(*) FROM grants").fetchone()[0])
                ceiling = self.ceilings[priority]
                if now < blocked_until:
                    granted = False
                    reason = "server-rate-limit-backoff"
                    retry_at = blocked_until
                elif count < ceiling:
                    self._connection.execute("INSERT INTO grants(granted_at_unix, operation, priority, process_id) VALUES (?, ?, ?, ?)", (now, operation, priority, os.getpid()))
                    count += 1
                    granted = True
                    reason = "granted"
                    retry_at = None
                else:
                    offset = count - ceiling
                    limiting = self._connection.execute("SELECT granted_at_unix FROM grants ORDER BY granted_at_unix, sequence LIMIT 1 OFFSET ?", (offset,)).fetchone()
                    granted = False
                    reason = "priority-ceiling-exhausted"
                    retry_at = float(limiting[0]) + self.window_s + 1e-6
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return {
            "contract": API_RATE_LIMITER_CONTRACT,
            "granted": granted,
            "operation": operation,
            "priority": priority,
            "reason": reason,
            "request_limit": self.request_limit,
            "priority_ceiling": ceiling,
            "requests_in_window": count,
            "remaining_at_priority": max(0, ceiling - count),
            "observed_at_unix": raw_now,
            "effective_at_unix": now,
            "clock_clamped": now != raw_now,
            "retry_at_unix": retry_at,
            "retry_after_s": max(0.0, float(retry_at) - now) if retry_at is not None else None,
        }

    def acquire(self, *, operation: str, priority: APIPriority, wait: bool = False, deadline_monotonic: float | None = None) -> dict[str, object]:
        started = self.monotonic()
        attempts = 0
        while True:
            attempts += 1
            receipt = self._attempt(operation=operation, priority=priority)
            now_monotonic = self.monotonic()
            if receipt["granted"]:
                return {**receipt, "attempts": attempts, "waited_s": round(max(0.0, now_monotonic - started), 6)}
            if not wait or (deadline_monotonic is not None and now_monotonic >= deadline_monotonic):
                reason = "deadline-exhausted" if wait and deadline_monotonic is not None and now_monotonic >= deadline_monotonic else receipt["reason"]
                return {**receipt, "reason": reason, "attempts": attempts, "waited_s": round(max(0.0, now_monotonic - started), 6)}
            sleep_s = max(0.001, float(receipt["retry_after_s"] or 0.001))
            if deadline_monotonic is not None:
                sleep_s = min(sleep_s, max(0.0, deadline_monotonic - now_monotonic))
                if sleep_s <= 0.0:
                    return {**receipt, "reason": "deadline-exhausted", "attempts": attempts, "waited_s": round(max(0.0, now_monotonic - started), 6)}
            self.sleeper(sleep_s)

    def record_server_rate_limit(self, *, operation: str, retry_after: object = None) -> dict[str, object]:
        now = float(self.clock())
        retry_after_s = _parse_retry_after(retry_after, now=now)
        blocked_until = now + retry_after_s
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                prior = float(self._connection.execute("SELECT blocked_until_unix FROM runtime_state WHERE singleton = 1").fetchone()[0])
                blocked_until = max(prior, blocked_until)
                self._connection.execute("UPDATE runtime_state SET blocked_until_unix = ? WHERE singleton = 1", (blocked_until,))
                self._connection.execute("INSERT INTO server_rate_limits(observed_at_unix, operation, retry_after_s, blocked_until_unix) VALUES (?, ?, ?, ?)", (now, operation, retry_after_s, blocked_until))
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return {"contract": API_RATE_LIMITER_CONTRACT, "operation": operation, "observed_at_unix": now, "retry_after_s": retry_after_s, "blocked_until_unix": blocked_until}

    def status(self) -> dict[str, object]:
        now = float(self.clock())
        cutoff = now - self.window_s
        with self._lock:
            rows = self._connection.execute("SELECT operation, priority, COUNT(*) FROM grants WHERE granted_at_unix > ? GROUP BY operation, priority ORDER BY operation, priority", (cutoff,)).fetchall()
            blocked_until = float(self._connection.execute("SELECT blocked_until_unix FROM runtime_state WHERE singleton = 1").fetchone()[0])
            rate_limits = int(self._connection.execute("SELECT COUNT(*) FROM server_rate_limits").fetchone()[0])
        return {
            **self.manifest_receipt(),
            "observed_at_unix": now,
            "requests_in_window": sum(int(row[2]) for row in rows),
            "requests_by_operation_priority": {f"{row[0]}:{row[1]}": int(row[2]) for row in rows},
            "blocked_until_unix": blocked_until,
            "blocked": now < blocked_until,
            "server_rate_limit_responses": rate_limits,
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()


@dataclass
class _RequestContext:
    operation: str
    priority: APIPriority
    wait: bool
    deadline_monotonic: float | None
    receipts: list[dict[str, object]] = field(default_factory=list)
    prepaid: list[dict[str, object]] = field(default_factory=list)


class _BudgetedSession:
    def __init__(self, *, session: Any, limiter: AgentWideGleeAPIRateLimiter, context: contextvars.ContextVar[_RequestContext | None]) -> None:
        self._session = session
        self._limiter = limiter
        self._context = context

    def request(self, method: str, url: str, **kwargs: object) -> Any:
        context = self._context.get()
        if context is None:
            raise GleeAPIBudgetBypass("authenticated GLEE request attempted outside the agent-wide budget gateway")
        receipt = context.prepaid.pop(0) if context.prepaid else self._limiter.acquire(operation=context.operation, priority=context.priority, wait=context.wait, deadline_monotonic=context.deadline_monotonic)
        context.receipts.append(receipt)
        if not receipt["granted"]:
            raise GleeAPIBudgetDeferred(receipt)
        response = self._session.request(method, url, **kwargs)
        if getattr(response, "status_code", None) == 429:
            headers = getattr(response, "headers", {})
            retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
            self._limiter.record_server_rate_limit(operation=context.operation, retry_after=retry_after)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


class AgentWideGleeAPIGateway:
    """Route every actual SDK HTTP attempt through one shared durable limiter."""

    def __init__(self, *, client: Any, limiter: AgentWideGleeAPIRateLimiter) -> None:
        self.client = client
        self.limiter = limiter
        self._context: contextvars.ContextVar[_RequestContext | None] = contextvars.ContextVar("glee_api_budget_context", default=None)
        session = getattr(client, "session", None)
        self._original_session = session if callable(getattr(session, "request", None)) else None
        self._budgeted_session = _BudgetedSession(session=session, limiter=limiter, context=self._context) if self._original_session is not None else None
        if self._budgeted_session is not None:
            client.session = self._budgeted_session

    @property
    def intercepts_sdk_retries(self) -> bool:
        return self._budgeted_session is not None

    def manifest_receipt(self) -> dict[str, object]:
        return {**self.limiter.manifest_receipt(), "sdk_retry_interception": self.intercepts_sdk_retries, "bypass_policy": "fail-closed"}

    def reserve(self, *, operation: str, priority: APIPriority, wait: bool = False, deadline_monotonic: float | None = None) -> dict[str, object]:
        receipt = self.limiter.acquire(operation=operation, priority=priority, wait=wait, deadline_monotonic=deadline_monotonic)
        if not receipt["granted"]:
            raise GleeAPIBudgetDeferred(receipt)
        return receipt

    def call(self, *, operation: str, priority: APIPriority, callback: Callable[[], Any], wait: bool = False, deadline_monotonic: float | None = None, prepaid: dict[str, object] | None = None) -> Any:
        if prepaid is not None and (prepaid.get("granted") is not True or prepaid.get("operation") != operation or prepaid.get("priority") != priority):
            raise ValueError("prepaid API grant does not match the requested operation and priority")
        context = _RequestContext(operation=operation, priority=priority, wait=wait, deadline_monotonic=deadline_monotonic, prepaid=[prepaid] if prepaid is not None else [])
        if self._budgeted_session is None:
            receipt = context.prepaid.pop(0) if context.prepaid else self.limiter.acquire(operation=operation, priority=priority, wait=wait, deadline_monotonic=deadline_monotonic)
            context.receipts.append(receipt)
            if not receipt["granted"]:
                raise GleeAPIBudgetDeferred(receipt)
            return callback()
        token = self._context.set(context)
        try:
            return callback()
        finally:
            self._context.reset(token)

    def close(self) -> None:
        if self._budgeted_session is not None and getattr(self.client, "session", None) is self._budgeted_session:
            self.client.session = self._original_session
