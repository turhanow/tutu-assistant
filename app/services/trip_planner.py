"""Application use case orchestrating reusable transport and hotel planning stages."""

from __future__ import annotations

import asyncio
from datetime import date

from app.domain.errors import ProviderError
from app.domain.models import (
    HotelMode,
    HotelOffer,
    HotelPreferences,
    HotelSearchQuery,
    TransportMode,
    TransportOffer,
    TransportSearchQuery,
    TripRequest,
    TripSearchFailure,
    TripSearchResult,
    TripTransportCandidates,
)
from app.ports.clock import Clock
from app.ports.tutu import TutuGateway
from app.services.combinations import build_combinations, required_hotel_stays
from app.services.date_rules import validate_planning_date
from app.services.ranking import rank_options


class TripPlanner:
    def __init__(
        self,
        gateway: TutuGateway,
        clock: Clock,
        *,
        timeout_seconds: int = 30,
    ) -> None:
        self._gateway = gateway
        self._clock = clock
        self._timeout_seconds = timeout_seconds

    async def search_trip(self, request: TripRequest) -> TripSearchResult:
        """Backward-compatible complete known-destination search."""
        now = self._clock.now()
        validate_planning_date(request, today=now.date())
        try:
            async with asyncio.timeout(self._timeout_seconds):
                candidates = await self._search_transport_candidates(request, searched_at=now)
                return await self._complete_trip(candidates, search_hotels=True)
        except TimeoutError:
            return self._failed_result(
                request,
                now,
                "search_timeout",
                "trip",
                True,
                "Поиск занял слишком много времени, попробуйте снова",
            )
        except ProviderError:
            return self._failed_result(
                request,
                now,
                "transport_provider_error",
                "transport",
                True,
                "Tutu временно не ответил по транспорту",
            )

    async def search_transport_candidates(
        self,
        request: TripRequest,
    ) -> TripTransportCandidates:
        """Search both legs once so discovery can reuse them for finalist hotel checks."""
        now = self._clock.now()
        validate_planning_date(request, today=now.date())
        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await self._search_transport_candidates(request, searched_at=now)
        except TimeoutError:
            return self._failed_transport_candidates(
                request,
                now,
                "transport_timeout",
                "Tutu не успел ответить по транспорту",
            )
        except ProviderError:
            return self._failed_transport_candidates(
                request,
                now,
                "transport_provider_error",
                "Tutu временно не ответил по транспорту",
            )

    async def complete_transport_candidates(
        self,
        candidates: TripTransportCandidates,
        *,
        search_hotels: bool,
    ) -> TripSearchResult:
        """Build ranked combinations, optionally enriching the reused transport with hotels."""
        if not candidates.is_feasible:
            return TripSearchResult(
                request=candidates.request,
                failures=candidates.failures,
                searched_at=candidates.searched_at,
            )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await self._complete_trip(candidates, search_hotels=search_hotels)
        except TimeoutError:
            return self._failed_result(
                candidates.request,
                candidates.searched_at,
                "hotel_timeout" if search_hotels else "planning_timeout",
                "hotel" if search_hotels else "trip",
                True,
                "Проверка поездки заняла слишком много времени",
            )
        except ProviderError:
            return self._failed_result(
                candidates.request,
                candidates.searched_at,
                "hotel_provider_error" if search_hotels else "planning_provider_error",
                "hotel" if search_hotels else "trip",
                True,
                "Tutu временно не ответил по размещению",
            )

    async def _search_transport_candidates(
        self,
        request: TripRequest,
        *,
        searched_at,
    ) -> TripTransportCandidates:
        outbound, return_offers = await asyncio.gather(
            self._search_leg(self._transport_query(request, outbound=True)),
            self._search_leg(self._transport_query(request, outbound=False)),
        )
        failures: tuple[TripSearchFailure, ...] = ()
        if not outbound or not return_offers:
            failures = (
                self._failure(
                    "no_transport",
                    "transport",
                    False,
                    "Подходящего транспорта в обе стороны не найдено",
                ),
            )
        return TripTransportCandidates(
            request=request,
            outbound=tuple(outbound),
            return_offers=tuple(return_offers),
            failures=failures,
            searched_at=searched_at,
        )

    async def _complete_trip(
        self,
        candidates: TripTransportCandidates,
        *,
        search_hotels: bool,
    ) -> TripSearchResult:
        request = candidates.request
        if not candidates.is_feasible:
            return TripSearchResult(
                request=request,
                failures=candidates.failures,
                searched_at=candidates.searched_at,
            )
        outbound = list(candidates.outbound)
        return_offers = list(candidates.return_offers)
        failures = list(candidates.failures)
        stays = required_hotel_stays(request, outbound, return_offers)
        hotels_by_stay: dict[tuple[date, date], list[HotelOffer]] = {}
        effective_request = request
        if not search_hotels:
            effective_request = request.model_copy(
                update={"hotel": HotelPreferences(mode=HotelMode.FORBIDDEN)}
            )
        elif request.hotel.mode is not HotelMode.FORBIDDEN:
            hotel_results = await asyncio.gather(
                *(self._search_hotels(request, stay) for stay in stays[:3]),
                return_exceptions=True,
            )
            for stay, hotel_result in zip(stays[:3], hotel_results, strict=True):
                if isinstance(hotel_result, ProviderError):
                    failures.append(
                        self._failure(
                            "hotel_provider_error",
                            "hotel",
                            True,
                            "Tutu временно не ответил по размещению",
                        )
                    )
                    continue
                if isinstance(hotel_result, BaseException):
                    raise hotel_result
                hotels_by_stay[(stay.check_in, stay.check_out)] = list(hotel_result)

        combinations = build_combinations(
            effective_request,
            outbound,
            return_offers,
            hotel_offers_by_stay=hotels_by_stay,
        )
        over_budget = False
        if not combinations and request.budget is not None:
            without_budget = build_combinations(
                effective_request.model_copy(update={"budget": None}),
                outbound,
                return_offers,
                hotel_offers_by_stay=hotels_by_stay,
            )
            exact_totals = [
                item.price.known_total
                for item in without_budget
                if item.price.known_total is not None
            ]
            if exact_totals:
                over_budget = True
                difference = min(exact_totals) - request.budget
                failures.append(
                    self._failure(
                        "over_budget",
                        "trip",
                        False,
                        "Ближайший полный вариант превышает бюджет "
                        f"на {difference.quantize(1)} {request.currency}",
                    )
                )
        if (
            search_hotels
            and request.hotel.mode is HotelMode.REQUIRED
            and not combinations
            and not over_budget
            and not stays
        ):
            failures.append(
                self._failure(
                    "no_feasible_pair",
                    "transport",
                    False,
                    "Предложения туда и обратно несовместимы по времени поездки",
                )
            )
        elif (
            search_hotels
            and request.hotel.mode is HotelMode.REQUIRED
            and not combinations
            and not over_budget
        ):
            failures.append(
                self._failure(
                    "no_hotel",
                    "hotel",
                    False,
                    "Размещение на все ночи не найдено",
                )
            )
        return TripSearchResult(
            request=request,
            options=tuple(rank_options(combinations, preferred=request.sort)),
            failures=tuple(failures),
            searched_at=candidates.searched_at,
        )

    async def _search_leg(self, query: TransportSearchQuery) -> list[TransportOffer]:
        offers = list(await self._gateway.search_transport(query))
        if offers or len(query.allowed_modes) == 1:
            return offers
        fallback_modes = (
            sorted(query.allowed_modes)
            if query.allowed_modes
            else [TransportMode.RAIL, TransportMode.AVIA]
        )
        for mode in fallback_modes[:2]:
            try:
                offers.extend(
                    await self._gateway.search_transport(
                        query.model_copy(update={"allowed_modes": frozenset({mode})})
                    )
                )
            except ProviderError:
                continue
            if offers:
                break
        return offers

    @staticmethod
    def _transport_query(request: TripRequest, *, outbound: bool) -> TransportSearchQuery:
        return TransportSearchQuery(
            origin=request.origin if outbound else request.destination,
            destination=request.destination if outbound else request.origin,
            departure_date=request.departure_date if outbound else request.return_date,
            adults=request.travelers.adults,
            allowed_modes=request.transport.allowed_modes,
            max_price=request.budget,
        )

    async def _search_hotels(self, request: TripRequest, stay) -> list[HotelOffer]:
        return await self._gateway.search_hotels(
            HotelSearchQuery(
                city=request.destination,
                check_in=stay.check_in,
                check_out=stay.check_out,
                adults=request.travelers.adults,
                rooms=request.travelers.rooms,
                preferences=request.hotel,
            )
        )

    @classmethod
    def _failed_result(
        cls,
        request: TripRequest,
        searched_at,
        category: str,
        component: str,
        retryable: bool,
        message: str,
    ) -> TripSearchResult:
        return TripSearchResult(
            request=request,
            failures=(cls._failure(category, component, retryable, message),),
            searched_at=searched_at,
        )

    @classmethod
    def _failed_transport_candidates(
        cls,
        request: TripRequest,
        searched_at,
        category: str,
        message: str,
    ) -> TripTransportCandidates:
        return TripTransportCandidates(
            request=request,
            failures=(cls._failure(category, "transport", True, message),),
            searched_at=searched_at,
        )

    @staticmethod
    def _failure(
        category: str,
        component: str,
        retryable: bool,
        message: str,
    ) -> TripSearchFailure:
        return TripSearchFailure(
            category=category,
            component=component,
            retryable=retryable,
            user_message=message,
        )
