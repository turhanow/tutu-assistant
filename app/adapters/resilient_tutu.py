"""Concurrency, rate-budget and circuit-breaker decorator for TutuGateway."""

from __future__ import annotations

import asyncio
import time
from collections import deque

from app.domain.errors import ProviderError, ProviderTransientError
from app.ports.tutu import TutuGateway
from app.services.resilience import CircuitBreaker


class ProtectedTutuGateway:
    def __init__(
        self,
        delegate: TutuGateway,
        *,
        max_inflight: int = 8,
        calls_per_minute: int = 120,
        failure_threshold: int = 3,
        recovery_seconds: float = 30,
    ) -> None:
        if max_inflight < 1 or calls_per_minute < 1:
            raise ValueError("provider limits must be positive")
        self._delegate = delegate
        self._semaphore = asyncio.Semaphore(max_inflight)
        self._calls_per_minute = calls_per_minute
        self._call_times: deque[float] = deque()
        self._budget_lock = asyncio.Lock()
        self._breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_seconds=recovery_seconds,
        )
        self._accepted_calls = 0
        self._rejected_calls = 0
        self._failed_calls = 0

    @property
    def circuit_state(self) -> str:
        return self._breaker.state.value

    @property
    def metrics(self) -> dict[str, int | str]:
        return {
            "accepted_calls": self._accepted_calls,
            "rejected_calls": self._rejected_calls,
            "failed_calls": self._failed_calls,
            "circuit_state": self.circuit_state,
        }

    async def discover_capabilities(self):
        return await self._execute(self._delegate.discover_capabilities)

    async def search_transport(self, query):
        return await self._execute(self._delegate.search_transport, query)

    async def search_hotels(self, query):
        return await self._execute(self._delegate.search_hotels, query)

    async def get_offer_details(self, ref):
        return await self._execute(self._delegate.get_offer_details, ref)

    async def create_checkout_link(self, ref):
        return await self._execute(self._delegate.create_checkout_link, ref)

    async def close(self) -> None:
        await self._delegate.close()

    async def _execute(self, operation, *args):
        try:
            self._breaker.before_call()
            await self._consume_budget()
        except ProviderTransientError:
            self._rejected_calls += 1
            raise
        self._accepted_calls += 1
        async with self._semaphore:
            try:
                result = await operation(*args)
            except ProviderError:
                self._failed_calls += 1
                self._breaker.record_failure()
                raise
            else:
                self._breaker.record_success()
                return result

    async def _consume_budget(self) -> None:
        now = time.monotonic()
        async with self._budget_lock:
            threshold = now - 60
            while self._call_times and self._call_times[0] <= threshold:
                self._call_times.popleft()
            if len(self._call_times) >= self._calls_per_minute:
                raise ProviderTransientError("provider call budget is exhausted")
            self._call_times.append(now)
