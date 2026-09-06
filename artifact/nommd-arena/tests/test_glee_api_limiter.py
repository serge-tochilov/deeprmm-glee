from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nommd_arena.glee_api_limiter import AgentWideGleeAPIGateway, AgentWideGleeAPIRateLimiter, GleeAPIBudgetBypass, GleeAPIBudgetDeferred


class FakeTime:
    def __init__(self, wall: float = 1_000.0, monotonic: float = 0.0) -> None:
        self.wall = wall
        self.monotonic = monotonic

    def sleep(self, seconds: float) -> None:
        self.wall += seconds
        self.monotonic += seconds


def _limiter(path: Path, clock: FakeTime, *, request_limit: int = 10, move_reserve: int = 2, control_reserve: int = 2) -> AgentWideGleeAPIRateLimiter:
    return AgentWideGleeAPIRateLimiter(state_path=path, request_limit=request_limit, window_s=60.0, move_reserve=move_reserve, control_reserve=control_reserve, clock=lambda: clock.wall, monotonic=lambda: clock.monotonic, sleeper=clock.sleep)


def test_priority_reserves_protect_control_and_critical_requests(tmp_path: Path) -> None:
    clock = FakeTime()
    limiter = _limiter(tmp_path / "budget.sqlite3", clock)
    try:
        assert all(limiter.acquire(operation="queue", priority="background")["granted"] for _ in range(6))
        assert limiter.acquire(operation="queue", priority="background")["granted"] is False
        assert all(limiter.acquire(operation="pending_games", priority="control")["granted"] for _ in range(2))
        assert limiter.acquire(operation="pending_games", priority="control")["granted"] is False
        assert all(limiter.acquire(operation="move", priority="critical")["granted"] for _ in range(2))
        assert limiter.acquire(operation="move", priority="critical")["granted"] is False
        status = limiter.status()
        assert status["requests_in_window"] == 10
        assert status["requests_by_operation_priority"] == {"move:critical": 2, "pending_games:control": 2, "queue:background": 6}
        clock.sleep(60.00001)
        assert limiter.acquire(operation="stats", priority="background")["granted"] is True
        assert limiter.status()["requests_in_window"] == 1
    finally:
        limiter.close()


def test_budget_is_restart_stable_and_shared_between_instances(tmp_path: Path) -> None:
    clock = FakeTime()
    path = tmp_path / "budget.sqlite3"
    first = _limiter(path, clock)
    second = _limiter(path, clock)
    try:
        assert first.acquire(operation="pending_games", priority="control")["granted"] is True
        assert second.status()["requests_in_window"] == 1
        assert second.acquire(operation="move", priority="critical")["granted"] is True
        assert first.status()["requests_in_window"] == 2
        assert path.stat().st_mode & 0o777 == 0o600
    finally:
        second.close()
        first.close()
    restarted = _limiter(path, clock)
    try:
        assert restarted.status()["requests_in_window"] == 2
    finally:
        restarted.close()


def test_concurrent_process_style_instances_cannot_oversubscribe_the_window(tmp_path: Path) -> None:
    clock = FakeTime()
    path = tmp_path / "budget.sqlite3"
    limiters = [_limiter(path, clock) for _ in range(20)]
    barrier = threading.Barrier(len(limiters))

    def acquire(limiter: AgentWideGleeAPIRateLimiter) -> bool:
        barrier.wait()
        return bool(limiter.acquire(operation="move", priority="critical")["granted"])

    try:
        with ThreadPoolExecutor(max_workers=len(limiters)) as pool:
            granted = list(pool.map(acquire, limiters))
        assert sum(granted) == 10
        assert limiters[0].status()["requests_in_window"] == 10
    finally:
        for limiter in limiters:
            limiter.close()


def test_critical_wait_respects_the_original_deadline(tmp_path: Path) -> None:
    clock = FakeTime()
    limiter = _limiter(tmp_path / "budget.sqlite3", clock, request_limit=3, move_reserve=0, control_reserve=0)
    try:
        for _ in range(3):
            assert limiter.acquire(operation="move", priority="critical")["granted"] is True
        denied = limiter.acquire(operation="move", priority="critical", wait=True, deadline_monotonic=30.0)
        assert denied["granted"] is False
        assert denied["reason"] == "deadline-exhausted"
        assert denied["waited_s"] == 30.0
        clock.sleep(30.00001)
        granted = limiter.acquire(operation="move", priority="critical", wait=True, deadline_monotonic=clock.monotonic + 1.0)
        assert granted["granted"] is True
    finally:
        limiter.close()


def test_gateway_budgets_every_underlying_sdk_retry_and_fails_closed_on_bypass(tmp_path: Path) -> None:
    class Response:
        status_code = 200
        headers: dict[str, str] = {}

    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def request(self, _method: str, _url: str, **_kwargs: object) -> Response:
            self.calls += 1
            return Response()

    class Client:
        def __init__(self) -> None:
            self.session = Session()

        def retried_get(self) -> int:
            for _ in range(3):
                self.session.request("GET", "https://example.invalid/api")
            return 3

    clock = FakeTime()
    limiter = _limiter(tmp_path / "budget.sqlite3", clock)
    client = Client()
    original_session = client.session
    gateway = AgentWideGleeAPIGateway(client=client, limiter=limiter)
    try:
        assert gateway.intercepts_sdk_retries is True
        prepaid = gateway.reserve(operation="pending_games", priority="control")
        assert gateway.call(operation="pending_games", priority="control", callback=client.retried_get, prepaid=prepaid) == 3
        assert limiter.status()["requests_in_window"] == 3
        with pytest.raises(GleeAPIBudgetBypass):
            client.session.request("GET", "https://example.invalid/api")
    finally:
        gateway.close()
        limiter.close()
    assert client.session is original_session


def test_server_429_blocks_every_process_until_retry_after(tmp_path: Path) -> None:
    class Response:
        status_code = 429
        headers = {"Retry-After": "5"}

    class Session:
        def request(self, _method: str, _url: str, **_kwargs: object) -> Response:
            return Response()

    class Client:
        def __init__(self) -> None:
            self.session = Session()

        def request(self) -> Response:
            return self.session.request("GET", "https://example.invalid/api")

    clock = FakeTime()
    path = tmp_path / "budget.sqlite3"
    first = _limiter(path, clock)
    client = Client()
    gateway = AgentWideGleeAPIGateway(client=client, limiter=first)
    try:
        response = gateway.call(operation="pending_games", priority="control", callback=client.request)
        assert response.status_code == 429
        second = _limiter(path, clock)
        try:
            denied = second.acquire(operation="move", priority="critical")
            assert denied["granted"] is False
            assert denied["reason"] == "server-rate-limit-backoff"
            assert second.status()["server_rate_limit_responses"] == 1
            clock.sleep(5.00001)
            assert second.acquire(operation="move", priority="critical")["granted"] is True
        finally:
            second.close()
    finally:
        gateway.close()
        first.close()


def test_direct_client_without_a_session_consumes_one_high_level_grant(tmp_path: Path) -> None:
    class Client:
        def stats(self) -> dict[str, int]:
            return {"active_games": 0}

    clock = FakeTime()
    limiter = _limiter(tmp_path / "budget.sqlite3", clock)
    gateway = AgentWideGleeAPIGateway(client=Client(), limiter=limiter)
    try:
        assert gateway.call(operation="stats", priority="background", callback=gateway.client.stats) == {"active_games": 0}
        assert limiter.status()["requests_by_operation_priority"] == {"stats:background": 1}
        for _ in range(5):
            gateway.call(operation="stats", priority="background", callback=gateway.client.stats)
        with pytest.raises(GleeAPIBudgetDeferred):
            gateway.call(operation="stats", priority="background", callback=gateway.client.stats)
    finally:
        gateway.close()
        limiter.close()
