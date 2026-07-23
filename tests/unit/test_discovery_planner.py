import asyncio
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

import pytest

from app.domain.content_models import DestinationProfile
from app.domain.discovery_models import (
    CandidateShortlist,
    DateFlexibility,
    DateRange,
    DestinationCandidate,
    DiscoveryRequest,
    ExperienceProfile,
    FeasibilityStatus,
    RoadTolerance,
    TravelPace,
)
from app.domain.models import (
    HotelMode,
    HotelOffer,
    OfferRef,
    TransportMode,
    TransportOffer,
)
from app.services.destination_feasibility import (
    DestinationFeasibilityService,
    resolve_search_dates,
)
from app.services.discovery_planner import DiscoveryPlanner
from app.services.trip_planner import TripPlanner


class FakeClock:
    def now(self) -> datetime:
        return datetime(2026, 7, 21, 12, tzinfo=UTC)


class MultiDestinationGateway:
    def __init__(
        self,
        *,
        unavailable: set[str] | None = None,
        hotels_unavailable: set[str] | None = None,
    ) -> None:
        self.unavailable = unavailable or set()
        self.hotels_unavailable = hotels_unavailable or set()
        self.transport_calls = []
        self.hotel_calls = []

    async def search_transport(self, query):
        self.transport_calls.append(query)
        city = query.destination if query.origin == "Москва" else query.origin
        if city in self.unavailable:
            return []
        departure_hour = 8 if query.origin == "Москва" else 18
        departure = datetime.combine(
            query.departure_date,
            time(departure_hour),
            tzinfo=UTC,
        )
        return [
            TransportOffer(
                provider_id=f"{query.origin}-{query.destination}-{query.departure_date}",
                mode=TransportMode.RAIL,
                origin=query.origin,
                destination=query.destination,
                departure_at=departure,
                arrival_at=departure + timedelta(hours=2),
                price="1000",
                currency="RUB",
                transfers=0,
                offer_ref=OfferRef(product_type="rail"),
            )
        ]

    async def search_hotels(self, query):
        self.hotel_calls.append(query)
        if query.city in self.hotels_unavailable:
            return []
        return [
            HotelOffer(
                hotel_id=f"hotel-{query.city}",
                offer_id=f"offer-{query.city}",
                name=f"Отель {query.city}",
                check_in=query.check_in,
                check_out=query.check_out,
                price="4000",
                currency="RUB",
                offer_ref=OfferRef(product_type="hotel"),
            )
        ]


def discovery_request(**overrides) -> DiscoveryRequest:
    values = {
        "origin": "Москва",
        "dates": DateRange(start=date(2026, 8, 22), end=date(2026, 8, 23)),
        "experience": ExperienceProfile(interests={"architecture"}),
    }
    values.update(overrides)
    return DiscoveryRequest(**values)


def candidate(index: int) -> DestinationCandidate:
    destination_id = f"city_{index}"
    return DestinationCandidate(
        destination=DestinationProfile(
            destination_id=destination_id,
            name=f"Город {index}",
            region=f"Регион {index}",
            experience_tags={"architecture"},
            evidence_ids={f"source_{index}"},
        ),
        match_score=Decimal("0.9") - Decimal(index) / Decimal(100),
        match_reasons=("Совпадает интерес к архитектуре",),
    )


def shortlist(count: int = 5, **request_overrides) -> CandidateShortlist:
    return CandidateShortlist(
        request=discovery_request(**request_overrides),
        candidates=tuple(candidate(index) for index in range(count)),
        catalog_version="v1",
        score_version="candidate_v1",
    )


def planner(gateway: MultiDestinationGateway) -> DiscoveryPlanner:
    trip_planner = TripPlanner(gateway, FakeClock())  # type: ignore[arg-type]
    feasibility = DestinationFeasibilityService(trip_planner)
    return DiscoveryPlanner(feasibility, FakeClock(), max_concurrency=3)


@pytest.mark.asyncio
async def test_discovery_without_hotel_never_calls_hotel_provider() -> None:
    gateway = MultiDestinationGateway()
    result = await planner(gateway).verify(
        shortlist(
            1,
            dates=DateRange(start=date(2026, 8, 22), end=date(2026, 8, 22)),
            hotel_mode=HotelMode.FORBIDDEN,
        )
    )

    assert result.snapshots
    assert not gateway.hotel_calls
    assert all(
        option.combination.hotel is None
        for snapshot in result.snapshots
        if snapshot.trip_result is not None
        for option in snapshot.trip_result.options
    )


def test_date_resolution_keeps_short_dates_and_selects_first_full_weekend() -> None:
    exact = discovery_request()
    assert resolve_search_dates(exact) == (date(2026, 8, 22), date(2026, 8, 23))

    flexible = discovery_request(
        dates=DateRange(
            start=date(2026, 8, 17),
            end=date(2026, 8, 30),
            flexibility=DateFlexibility.RANGE,
        )
    )
    assert resolve_search_dates(flexible) == (date(2026, 8, 22), date(2026, 8, 23))


def test_fixed_long_window_is_not_silently_reinterpreted() -> None:
    request = discovery_request(
        dates=DateRange(
            start=date(2026, 8, 17),
            end=date(2026, 8, 30),
            flexibility=DateFlexibility.FIXED,
        )
    )

    with pytest.raises(ValueError, match="fixed"):
        resolve_search_dates(request)


@pytest.mark.asyncio
async def test_five_transport_checks_reuse_offers_and_only_three_destinations_get_hotels() -> None:
    gateway = MultiDestinationGateway()

    result = await planner(gateway).verify(shortlist())

    assert len(gateway.transport_calls) == 10
    assert len(gateway.hotel_calls) == 3
    assert [item.status for item in result.snapshots] == [
        FeasibilityStatus.VERIFIED,
        FeasibilityStatus.VERIFIED,
        FeasibilityStatus.VERIFIED,
        FeasibilityStatus.TRANSPORT_ONLY,
        FeasibilityStatus.TRANSPORT_ONLY,
    ]
    assert len({item.destination_id for item in result.snapshots}) == 5


@pytest.mark.asyncio
async def test_one_unavailable_destination_does_not_cancel_other_results() -> None:
    gateway = MultiDestinationGateway(unavailable={"Город 1"})

    result = await planner(gateway).verify(shortlist())

    by_id = {item.destination_id: item for item in result.snapshots}
    assert by_id["city_1"].status is FeasibilityStatus.UNAVAILABLE
    assert by_id["city_0"].status is FeasibilityStatus.VERIFIED
    assert sum(item.status is FeasibilityStatus.VERIFIED for item in result.snapshots) == 3
    assert len(gateway.hotel_calls) == 3


@pytest.mark.asyncio
async def test_planner_checks_later_candidates_until_three_complete_options_exist() -> None:
    gateway = MultiDestinationGateway(
        hotels_unavailable={"Город 0", "Город 1"},
    )

    result = await planner(gateway).verify(shortlist(count=6))

    verified = [
        item for item in result.snapshots if item.status is FeasibilityStatus.VERIFIED
    ]
    assert len(verified) >= 3
    assert len(gateway.hotel_calls) == 6
    assert {"city_2", "city_3", "city_4"}.issubset(
        {item.destination_id for item in verified}
    )


@pytest.mark.asyncio
async def test_invalid_fixed_window_becomes_typed_unavailable_snapshot() -> None:
    gateway = MultiDestinationGateway()
    invalid = shortlist(
        count=1,
        dates=DateRange(
            start=date(2026, 8, 17),
            end=date(2026, 8, 30),
            flexibility=DateFlexibility.FIXED,
        ),
    )

    result = await planner(gateway).verify(invalid)

    assert result.snapshots[0].status is FeasibilityStatus.UNAVAILABLE
    assert result.snapshots[0].failures[0].category == "unsupported_discovery_dates"
    assert not gateway.transport_calls
    assert not gateway.hotel_calls


@pytest.mark.asyncio
async def test_discovery_overall_timeout_returns_recoverable_snapshots() -> None:
    class SlowGateway(MultiDestinationGateway):
        async def search_transport(self, query):
            await asyncio.sleep(0.05)
            return await super().search_transport(query)

    gateway = SlowGateway()
    trip_planner = TripPlanner(gateway, FakeClock())  # type: ignore[arg-type]
    discovery = DiscoveryPlanner(
        DestinationFeasibilityService(trip_planner),
        FakeClock(),
        timeout_seconds=0.001,
    )

    result = await discovery.verify(shortlist(count=2))

    assert all(item.status is FeasibilityStatus.UNAVAILABLE for item in result.snapshots)
    assert all(item.failures[0].category == "discovery_timeout" for item in result.snapshots)


@pytest.mark.asyncio
async def test_same_day_discovery_never_calls_hotel_provider() -> None:
    gateway = MultiDestinationGateway()
    same_day = shortlist(
        count=2,
        dates=DateRange(start=date(2026, 8, 22), end=date(2026, 8, 22)),
    )

    result = await planner(gateway).verify(same_day)

    assert all(item.status is FeasibilityStatus.VERIFIED for item in result.snapshots)
    assert not gateway.hotel_calls


@pytest.mark.asyncio
async def test_no_night_constraint_filters_overnight_transport_before_hotel_calls() -> None:
    class OvernightGateway(MultiDestinationGateway):
        async def search_transport(self, query):
            self.transport_calls.append(query)
            if query.origin == "Москва":
                departure = datetime.combine(query.departure_date, time(22, 50), tzinfo=UTC)
                duration = timedelta(hours=3)
            else:
                departure = datetime.combine(query.departure_date, time(7, 30), tzinfo=UTC)
                duration = timedelta(hours=2)
            return [
                TransportOffer(
                    provider_id=f"overnight-{query.origin}",
                    mode=TransportMode.RAIL,
                    origin=query.origin,
                    destination=query.destination,
                    departure_at=departure,
                    arrival_at=departure + duration,
                    price="1000",
                    currency="RUB",
                    transfers=0,
                    offer_ref=OfferRef(product_type="rail"),
                )
            ]

    gateway = OvernightGateway()
    result = await planner(gateway).verify(
        shortlist(
            1,
            hotel_mode=HotelMode.REQUIRED,
            experience=ExperienceProfile(
                interests={"river", "walking"},
                road_tolerance=RoadTolerance(allow_night_travel=False),
            ),
        )
    )

    assert result.snapshots[0].status is FeasibilityStatus.UNAVAILABLE
    assert result.snapshots[0].failures[0].category == "road_constraints_no_match"
    assert not gateway.hotel_calls


@pytest.mark.asyncio
async def test_relaxed_pace_rejects_short_city_window_without_night_legs() -> None:
    class RushedGateway(MultiDestinationGateway):
        async def search_transport(self, query):
            self.transport_calls.append(query)
            departure_hour = 20 if query.origin == "Москва" else 6
            departure_minute = 30
            departure = datetime.combine(
                query.departure_date,
                time(departure_hour, departure_minute),
                tzinfo=UTC,
            )
            return [
                TransportOffer(
                    provider_id=f"rushed-{query.origin}",
                    mode=TransportMode.RAIL,
                    origin=query.origin,
                    destination=query.destination,
                    departure_at=departure,
                    arrival_at=departure + timedelta(hours=2),
                    price="1000",
                    currency="RUB",
                    transfers=0,
                    offer_ref=OfferRef(product_type="rail"),
                )
            ]

    gateway = RushedGateway()
    result = await planner(gateway).verify(
        shortlist(
            1,
            experience=ExperienceProfile(
                interests={"history", "walking"},
                pace=TravelPace.RELAXED,
            ),
        )
    )

    assert result.snapshots[0].status is FeasibilityStatus.UNAVAILABLE
    assert result.snapshots[0].failures[-1].category == "pace_constraints_no_match"
    assert not gateway.hotel_calls


@pytest.mark.asyncio
async def test_destination_level_concurrency_is_bounded() -> None:
    gateway = MultiDestinationGateway()
    delegate = DestinationFeasibilityService(
        TripPlanner(gateway, FakeClock())  # type: ignore[arg-type]
    )

    class TrackingFeasibility:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def prepare(self, request, item):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.01)
                return await delegate.prepare(request, item)
            finally:
                self.active -= 1

        async def verify(self, prepared):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.01)
                return await delegate.verify(prepared)
            finally:
                self.active -= 1

        @staticmethod
        def transport_snapshot(prepared):
            return delegate.transport_snapshot(prepared)

    tracking = TrackingFeasibility()
    discovery = DiscoveryPlanner(
        tracking,  # type: ignore[arg-type]
        FakeClock(),
        max_concurrency=2,
    )

    await discovery.verify(shortlist())

    assert tracking.max_active == 2
