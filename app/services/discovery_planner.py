"""Bounded multi-destination transport and hotel feasibility orchestration."""

from __future__ import annotations

import asyncio

from app.domain.discovery_models import (
    CandidateShortlist,
    DiscoveryFeasibilityResult,
    FeasibilitySnapshot,
    FeasibilityStatus,
)
from app.domain.models import TripSearchFailure
from app.ports.clock import Clock
from app.services.destination_feasibility import (
    DestinationFeasibilityService,
    DestinationPreparation,
)

MAX_DESTINATIONS = 5
MAX_HOTEL_DESTINATIONS = 3


class DiscoveryPlanner:
    def __init__(
        self,
        feasibility: DestinationFeasibilityService,
        clock: Clock,
        *,
        max_concurrency: int = 3,
        timeout_seconds: int = 60,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._feasibility = feasibility
        self._clock = clock
        self._max_concurrency = max_concurrency
        self._timeout_seconds = timeout_seconds

    async def verify(self, shortlist: CandidateShortlist) -> DiscoveryFeasibilityResult:
        candidates = shortlist.candidates[:MAX_DESTINATIONS]
        if not candidates:
            return DiscoveryFeasibilityResult(
                shortlist=shortlist,
                snapshots=(),
                completed_at=self._clock.now(),
            )
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def prepare(candidate) -> DestinationPreparation:
            async with semaphore:
                return await self._feasibility.prepare(shortlist.request, candidate)

        async def verify(prepared: DestinationPreparation) -> FeasibilitySnapshot:
            async with semaphore:
                return await self._feasibility.verify(prepared)

        try:
            async with asyncio.timeout(self._timeout_seconds):
                prepared_items = await asyncio.gather(*(prepare(item) for item in candidates))
                finalists = [item for item in prepared_items if item.is_transport_feasible][
                    :MAX_HOTEL_DESTINATIONS
                ]
                verified = await asyncio.gather(*(verify(item) for item in finalists))
                verified_by_id = {item.destination_id: item for item in verified}
                snapshots: list[FeasibilitySnapshot] = []
                for prepared in prepared_items:
                    destination_id = prepared.candidate.destination.destination_id
                    if destination_id in verified_by_id:
                        snapshots.append(verified_by_id[destination_id])
                    elif prepared.preliminary_result is not None:
                        snapshots.append(self._feasibility.transport_snapshot(prepared))
                    else:
                        snapshots.append(self._failed_preparation_snapshot(prepared))
                return DiscoveryFeasibilityResult(
                    shortlist=shortlist,
                    snapshots=tuple(snapshots),
                    completed_at=self._clock.now(),
                )
        except TimeoutError:
            return DiscoveryFeasibilityResult(
                shortlist=shortlist,
                snapshots=tuple(self._timeout_snapshot(item) for item in candidates),
                completed_at=self._clock.now(),
            )

    def _failed_preparation_snapshot(
        self,
        prepared: DestinationPreparation,
    ) -> FeasibilitySnapshot:
        return FeasibilitySnapshot(
            destination_id=prepared.candidate.destination.destination_id,
            status=FeasibilityStatus.UNAVAILABLE,
            verified_at=self._clock.now(),
            failures=prepared.failures,
        )

    def _timeout_snapshot(self, candidate) -> FeasibilitySnapshot:
        return FeasibilitySnapshot(
            destination_id=candidate.destination.destination_id,
            status=FeasibilityStatus.UNAVAILABLE,
            verified_at=self._clock.now(),
            failures=(
                TripSearchFailure(
                    category="discovery_timeout",
                    component="trip",
                    retryable=True,
                    user_message="Проверка направлений заняла слишком много времени",
                ),
            ),
        )
