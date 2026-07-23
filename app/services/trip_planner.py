"""Application use case orchestrating reusable transport and hotel planning stages."""

from __future__ import annotations

import asyncio
from datetime import date

from app.domain.errors import ProviderError
from app.domain.models import (
    HotelMode,
    HotelOffer,
    HotelSearchQuery,
    TransportMode,
    TransportOffer,
    TransportPreferences,
    TransportSearchQuery,
    TripRequest,
    TripSearchFailure,
    TripSearchResult,
    TripTransportCandidates,
)
from app.ports.clock import Clock
from app.ports.tutu import TutuGateway
from app.services.combinations import (
    MAX_COMBINATIONS,
    MAX_COMBINATIONS_RELAXED,
    MAX_TRANSPORT_OFFERS_PER_LEG,
    MAX_TRANSPORT_PAIRS,
    build_combinations,
    required_hotel_stays,
)
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
        hotels_by_stay: dict[tuple[date, date], list[HotelOffer]] = {}
        effective_request = request
        requires_hotel = request.hotel.mode is HotelMode.REQUIRED
        combination_requests = tuple(self._combination_request_stages(effective_request))
        combination_limits = self._combination_limit_stages()
        combinations: list = []
        known_signatures = set()

        for stage_request, stage_warning in combination_requests:
            for (
                max_offers_per_leg,
                max_transport_pairs,
                max_combinations,
            ) in combination_limits:
                if combinations and len(combinations) >= 3:
                    break

                if search_hotels and stage_request.hotel.mode is not HotelMode.FORBIDDEN:
                    stays = required_hotel_stays(
                        stage_request,
                        outbound,
                        return_offers,
                        max_transport_offers_per_leg=max_offers_per_leg,
                        max_transport_pairs=max_transport_pairs,
                    )
                    hotel_results = await asyncio.gather(
                        *(
                            self._search_hotels(stage_request, stay)
                            for stay in stays[:3]
                            if (stay.check_in, stay.check_out) not in hotels_by_stay
                        ),
                        return_exceptions=True,
                    )
                    for stay, hotel_result in zip(
                        [
                            stay
                            for stay in stays[:3]
                            if (stay.check_in, stay.check_out) not in hotels_by_stay
                        ],
                        hotel_results,
                        strict=True,
                    ):
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

                built = build_combinations(
                    stage_request,
                    outbound,
                    return_offers,
                    hotel_offers_by_stay=hotels_by_stay,
                    allow_transport_only=not search_hotels,
                    max_transport_offers_per_leg=max_offers_per_leg,
                    max_transport_pairs=max_transport_pairs,
                    max_combinations=max_combinations,
                )
                for option in built:
                    if option.signature in known_signatures:
                        continue
                    known_signatures.add(option.signature)
                    if stage_warning:
                        option = option.model_copy(
                            update={"warnings": (*option.warnings, stage_warning)}
                        )
                    combinations.append(option)
                    if len(combinations) >= 3:
                        break
                if len(combinations) >= 3:
                    break
            if len(combinations) >= 3:
                break
        over_budget = False
        if not combinations and request.budget is not None:
            without_budget_request = effective_request.model_copy(update={"budget": None})
            without_budget = build_combinations(
                without_budget_request,
                outbound,
                return_offers,
                hotel_offers_by_stay=hotels_by_stay,
                max_transport_offers_per_leg=MAX_TRANSPORT_OFFERS_PER_LEG + 20,
                max_transport_pairs=MAX_TRANSPORT_PAIRS * 4,
                max_combinations=MAX_COMBINATIONS_RELAXED,
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
        if search_hotels and requires_hotel and not combinations and not over_budget:
            required_stays = required_hotel_stays(
                request,
                outbound,
                return_offers,
                max_transport_offers_per_leg=MAX_TRANSPORT_OFFERS_PER_LEG + 20,
                max_transport_pairs=MAX_TRANSPORT_PAIRS * 4,
            )
            if required_stays:
                failures.append(
                    self._failure(
                        "no_hotel",
                        "hotel",
                        False,
                        "Размещение на все ночи не найдено",
                    )
                )
            else:
                failures.append(
                    self._failure(
                        "no_feasible_pair",
                        "transport",
                        False,
                        "Предложения туда и обратно несовместимы по времени поездки",
                    )
                )
        return TripSearchResult(
            request=request,
            options=tuple(rank_options(combinations, preferred=request.sort)),
            failures=tuple(failures),
            searched_at=candidates.searched_at,
        )

    @staticmethod
    def _combination_limit_stages() -> tuple[tuple[int, int, int], ...]:
        return (
            (MAX_TRANSPORT_OFFERS_PER_LEG, MAX_TRANSPORT_PAIRS, MAX_COMBINATIONS),
            (MAX_TRANSPORT_OFFERS_PER_LEG, MAX_TRANSPORT_PAIRS, MAX_COMBINATIONS_RELAXED),
            (MAX_TRANSPORT_OFFERS_PER_LEG + 10, MAX_TRANSPORT_PAIRS, MAX_COMBINATIONS_RELAXED),
            (
                MAX_TRANSPORT_OFFERS_PER_LEG + 10,
                MAX_TRANSPORT_PAIRS * 2,
                MAX_COMBINATIONS_RELAXED,
            ),
            (
                MAX_TRANSPORT_OFFERS_PER_LEG + 20,
                MAX_TRANSPORT_PAIRS * 4,
                MAX_COMBINATIONS_RELAXED,
            ),
        )

    def _combination_request_stages(
        self,
        request: TripRequest,
    ) -> tuple[tuple[TripRequest, str | None], ...]:
        base = (request, None)
        has_windows = (
            request.transport.departure_window is not None
            or request.transport.return_window is not None
        )
        has_transfers = request.transport.max_transfers is not None
        variants: list[tuple[TripRequest, str | None]] = [base]
        if has_windows:
            relaxed_window = request.model_copy(
                update={
                    "transport": TransportPreferences(
                        allowed_modes=request.transport.allowed_modes,
                        max_transfers=request.transport.max_transfers,
                    )
                }
            )
            variants.append((relaxed_window, "Время отправления выходит за предпочитаемое окно."))
        if has_transfers:
            relaxed_transfers = request.model_copy(
                update={
                    "transport": TransportPreferences(
                        allowed_modes=request.transport.allowed_modes,
                        departure_window=request.transport.departure_window,
                        return_window=request.transport.return_window,
                        max_transfers=None,
                    )
                }
            )
            variants.append((relaxed_transfers, "Ограничение на количество пересадок ослаблено."))
        if has_windows and has_transfers:
            variants.append(
                (
                    request.model_copy(
                        update={
                            "transport": TransportPreferences(
                                allowed_modes=request.transport.allowed_modes,
                            )
                        }
                    ),
                    "Поиск идёт с расширенными параметрами маршрута.",
                )
            )
        seen: dict[str, str | None] = {}
        ordered: list[tuple[TripRequest, str | None]] = []
        for request_variant, reason in variants:
            key = repr(
                (
                    request_variant.transport.allowed_modes,
                    request_variant.transport.max_transfers,
                    request_variant.transport.departure_window,
                    request_variant.transport.return_window,
                )
            )
            if key not in seen:
                seen[key] = reason
                ordered.append((request_variant, reason))
        return tuple(ordered)

    async def _search_leg(self, query: TransportSearchQuery) -> list[TransportOffer]:
        offers = self._offers_with_exact_checkout(list(await self._gateway.search_transport(query)))
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
                    self._offers_with_exact_checkout(
                        await self._gateway.search_transport(
                            query.model_copy(update={"allowed_modes": frozenset({mode})})
                        )
                    )
                )
            except ProviderError:
                continue
            if offers:
                break
        return offers

    @staticmethod
    def _offers_with_exact_checkout(
        offers: list[TransportOffer] | tuple[TransportOffer, ...],
    ) -> list[TransportOffer]:
        """Exclude modes for which Tutu cannot open the selected departure for checkout."""
        unsupported_modes = {
            TransportMode.ETRAIN,
            TransportMode.MIXED,
            TransportMode.UNKNOWN,
        }
        return [
            offer
            for offer in offers
            if offer.mode not in unsupported_modes
            and not (offer.mode is TransportMode.AVIA and offer.transfers != 0)
        ]

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
