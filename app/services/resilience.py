"""Provider circuit breaker with deterministic state transitions."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

from app.domain.errors import ProviderTransientError


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_seconds: float = 30,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1 or recovery_seconds <= 0:
            raise ValueError("invalid circuit breaker limits")
        self._threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._monotonic = monotonic
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_inflight = False

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._monotonic() - self._opened_at >= self._recovery_seconds:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def before_call(self) -> None:
        state = self.state
        if state is CircuitState.OPEN:
            raise ProviderTransientError("provider circuit breaker is open")
        if state is CircuitState.HALF_OPEN:
            if self._half_open_inflight:
                raise ProviderTransientError("provider circuit breaker probe is in progress")
            self._half_open_inflight = True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._half_open_inflight = False

    def record_failure(self) -> None:
        self._half_open_inflight = False
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = self._monotonic()
