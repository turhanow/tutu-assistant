"""Prepare and verify one catalog destination using reusable TripPlanner stages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from app.domain.discovery_models import (
    DateFlexibility,
    DestinationCandidate,
    DiscoveryRequest,
    FeasibilitySnapshot,
    FeasibilityStatus,
    TravelPace,
)
from app.domain.models import (
    HotelMode,
    HotelPreferences,
    TransportOffer,
    TripRequest,
    TripSearchFailure,
    TripSearchResult,
    TripTransportCandidates,
)
from app.services.date_rules import night_travel_hours
from app.services.trip_planner import TripPlanner


@dataclass(frozen=True, slots=True)
class DestinationPreparation:
    candidate: DestinationCandidate
    discovery_request: DiscoveryRequest
    trip_request: TripRequest | None
    transport_candidates: TripTransportCandidates | None
    preliminary_result: TripSearchResult | None
    failures: tuple[TripSearchFailure, ...] = ()

    @property
    def is_transport_feasible(self) -> bool:
        return self.preliminary_result is not None and bool(self.preliminary_result.options)


class DestinationFeasibilityService:
    def __init__(self, planner: TripPlanner) -> None:
        self._planner = planner

    async def prepare(
        self,
        request: DiscoveryRequest,
        candidate: DestinationCandidate,
    ) -> DestinationPreparation:
        try:
            departure_date, return_date = resolve_search_dates(request)
            trip_request = TripRequest(
                origin=request.origin,
                destination=candidate.destination.name,
                departure_date=departure_date,
                return_date=return_date,
                travelers=request.travelers,
                budget=request.budget,
                currency=request.currency,
                transport=request.transport,
                hotel=HotelPreferences(
                    mode=(
                        HotelMode.FORBIDDEN if departure_date == return_date else request.hotel_mode
                    )
                ),
            )
        except ValueError:
            failure = TripSearchFailure(
                category="unsupported_discovery_dates",
                component="dates",
                retryable=False,
                user_message="Для проверки направления нужны конкретные даты до четырёх дней",
            )
            return DestinationPreparation(
                candidate=candidate,
                discovery_request=request,
                trip_request=None,
                transport_candidates=None,
                preliminary_result=None,
                failures=(failure,),
            )
        transport = await self._planner.search_transport_candidates(trip_request)
        transport = _filter_transport_candidates(request, transport)
        preliminary = await self._planner.complete_transport_candidates(
            transport,
            search_hotels=False,
        )
        preliminary = _filter_trip_result(request, preliminary)
        return DestinationPreparation(
            candidate=candidate,
            discovery_request=request,
            trip_request=trip_request,
            transport_candidates=transport,
            preliminary_result=preliminary,
            failures=preliminary.failures,
        )

    async def verify(self, prepared: DestinationPreparation) -> FeasibilitySnapshot:
        if prepared.transport_candidates is None or prepared.trip_request is None:
            return self.transport_snapshot(prepared)
        result = await self._planner.complete_transport_candidates(
            prepared.transport_candidates,
            search_hotels=True,
        )
        result = _filter_trip_result(prepared.discovery_request, result)
        if result.options and result.failures:
            status = FeasibilityStatus.PARTIAL
        elif result.options:
            status = FeasibilityStatus.VERIFIED
        else:
            status = FeasibilityStatus.UNAVAILABLE
        return FeasibilitySnapshot(
            destination_id=prepared.candidate.destination.destination_id,
            status=status,
            verified_at=result.searched_at,
            trip_result=result,
            failures=result.failures,
        )

    @staticmethod
    def transport_snapshot(prepared: DestinationPreparation) -> FeasibilitySnapshot:
        result = prepared.preliminary_result
        if result is not None and result.options:
            return FeasibilitySnapshot(
                destination_id=prepared.candidate.destination.destination_id,
                status=FeasibilityStatus.TRANSPORT_ONLY,
                verified_at=result.searched_at,
                trip_result=result,
                failures=result.failures,
            )
        searched_at = (
            result.searched_at
            if result is not None
            else prepared.transport_candidates.searched_at
            if prepared.transport_candidates is not None
            else None
        )
        if searched_at is None:
            # Invalid date preparations happen before provider calls; use a stable aware time
            # from the candidate request path in the orchestrator instead of fabricating one.
            raise ValueError("failed preparation requires orchestrator timestamp")
        return FeasibilitySnapshot(
            destination_id=prepared.candidate.destination.destination_id,
            status=FeasibilityStatus.UNAVAILABLE,
            verified_at=searched_at,
            trip_result=result,
            failures=prepared.failures,
        )


def resolve_search_dates(request: DiscoveryRequest) -> tuple[date, date]:
    start = request.dates.start
    end = request.dates.end
    if (end - start).days <= 3:
        return start, end
    if request.dates.flexibility not in {
        DateFlexibility.RANGE,
        DateFlexibility.WEEKEND,
    }:
        raise ValueError("fixed discovery dates exceed supported trip duration")
    cursor = start
    while cursor <= end:
        if cursor.weekday() == 5 and cursor + timedelta(days=1) <= end:
            return cursor, cursor + timedelta(days=1)
        cursor += timedelta(days=1)
    raise ValueError("date range does not contain a complete weekend")


def _filter_transport_candidates(
    request: DiscoveryRequest,
    candidates: TripTransportCandidates,
) -> TripTransportCandidates:
    """Apply explicit comfort constraints before hotel calls or proposal ranking."""
    outbound = tuple(
        offer for offer in candidates.outbound if _transport_offer_allowed(request, offer)
    )
    return_offers = tuple(
        offer for offer in candidates.return_offers if _transport_offer_allowed(request, offer)
    )
    failures = candidates.failures
    if candidates.is_feasible and (not outbound or not return_offers):
        failures = (
            *failures,
            TripSearchFailure(
                category="road_constraints_no_match",
                component="transport",
                retryable=False,
                user_message="Нет транспорта, который соблюдает ограничения по дороге",
            ),
        )
    return candidates.model_copy(
        update={
            "outbound": outbound,
            "return_offers": return_offers,
            "failures": failures,
        }
    )


def _transport_offer_allowed(request: DiscoveryRequest, offer: TransportOffer) -> bool:
    tolerance = request.experience.road_tolerance
    duration = offer.duration or (offer.arrival_at - offer.departure_at)
    if tolerance.max_one_way_duration is not None and duration > tolerance.max_one_way_duration:
        return False
    night_forbidden = (
        tolerance.allow_night_travel is False or request.experience.pace is TravelPace.RELAXED
    )
    return not night_forbidden or night_travel_hours(offer) == 0


def _filter_trip_result(
    request: DiscoveryRequest,
    result: TripSearchResult,
) -> TripSearchResult:
    options = result.options
    if request.experience.pace is TravelPace.RELAXED:
        calendar_days = (request.dates.end - request.dates.start).days + 1
        minimum_city_time = timedelta(hours=6 * calendar_days)
        options = tuple(
            option
            for option in options
            if option.combination.metrics.night_travel_hours == 0
            and option.combination.metrics.time_in_city >= minimum_city_time
        )
    failures = result.failures
    if result.options and not options:
        failures = (
            *failures,
            TripSearchFailure(
                category="pace_constraints_no_match",
                component="trip",
                retryable=False,
                user_message=(
                    "Найденные варианты слишком утомительные для спокойного темпа: "
                    "мало времени в городе или есть ночная дорога"
                ),
            ),
        )
    return result.model_copy(update={"options": options, "failures": failures})
