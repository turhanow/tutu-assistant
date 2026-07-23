import json
from datetime import UTC, date, datetime, time
from pathlib import Path

import pytest

from app.adapters.mcp_mappers import map_hotel_search, map_transport_search
from app.domain.models import (
    HotelMode,
    HotelPreferences,
    TimeWindow,
    TransportMode,
    TransportPreferences,
    TripRequest,
    TutuCapabilities,
)
from app.services.trip_planner import TripPlanner
from tests.fakes.tutu import FakeTutuGateway

FIXTURES = Path("tests/fixtures/tutu_mcp/fixtures")


class FakeClock:
    def now(self) -> datetime:
        return datetime(2026, 7, 21, tzinfo=UTC)


def payload(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def inputs():
    outbound = map_transport_search(payload("transport-search.json"))[0]
    return_offer = outbound.model_copy(
        update={
            "provider_id": "return-fixture",
            "origin": "Казань",
            "destination": "Москва",
            "departure_at": datetime.fromisoformat("2026-08-23T18:00:00+03:00"),
            "arrival_at": datetime.fromisoformat("2026-08-24T06:00:00+03:00"),
        }
    )
    hotel = map_hotel_search(payload("hotel-search.json"))[0]
    hotel = hotel.model_copy(update={"check_in": date(2026, 8, 22), "check_out": date(2026, 8, 23)})
    capabilities = TutuCapabilities(
        tools=(),
        schema_hash="fixture",
        captured_at=datetime(2026, 7, 21, tzinfo=UTC),
    )
    request = TripRequest(
        origin="Москва",
        destination="Казань",
        departure_date="2026-08-21",
        return_date="2026-08-23",
        hotel=HotelPreferences(mode=HotelMode.REQUIRED),
    )
    return request, outbound, return_offer, hotel, capabilities


@pytest.mark.asyncio
async def test_fake_end_to_end_returns_ranked_option_and_call_trace() -> None:
    request, outbound, return_offer, hotel, capabilities = inputs()
    gateway = FakeTutuGateway(
        capabilities=capabilities,
        transport=[outbound],
        hotels=[hotel],
    )
    transport_calls = 0

    async def directional_transport(query):
        nonlocal transport_calls
        transport_calls += 1
        gateway.calls.append(("search_transport", query))
        return [outbound] if transport_calls == 1 else [return_offer]

    gateway.search_transport = directional_transport
    result = await TripPlanner(gateway, FakeClock()).search_trip(request)

    assert len(result.options) == 1
    assert not result.failures
    assert [name for name, _ in gateway.calls] == [
        "search_transport",
        "search_transport",
        "search_hotels",
    ]


@pytest.mark.asyncio
async def test_no_transport_returns_typed_failure_without_hotel_call() -> None:
    request, _, _, _, capabilities = inputs()
    gateway = FakeTutuGateway(capabilities=capabilities)
    result = await TripPlanner(gateway, FakeClock()).search_trip(request)

    assert not result.options
    assert result.failures[0].category == "no_transport"
    assert all(name != "search_hotels" for name, _ in gateway.calls)


@pytest.mark.asyncio
async def test_required_hotel_empty_returns_no_complete_option() -> None:
    request, outbound, return_offer, _, capabilities = inputs()
    gateway = FakeTutuGateway(capabilities=capabilities, transport=[outbound])
    calls = 0

    async def directional_transport(_):
        nonlocal calls
        calls += 1
        return [outbound] if calls == 1 else [return_offer]

    gateway.search_transport = directional_transport
    result = await TripPlanner(gateway, FakeClock()).search_trip(request)

    assert not result.options
    assert result.failures[0].category == "no_hotel"


@pytest.mark.asyncio
async def test_empty_multitransport_uses_bounded_targeted_fallback() -> None:
    request, outbound, return_offer, hotel, capabilities = inputs()
    gateway = FakeTutuGateway(capabilities=capabilities, hotels=[hotel])

    async def fallback_transport(query):
        gateway.calls.append(("search_transport", query))
        if not query.allowed_modes:
            return []
        assert query.allowed_modes == {TransportMode.RAIL}
        return [outbound] if query.origin == "Москва" else [return_offer]

    gateway.search_transport = fallback_transport
    result = await TripPlanner(gateway, FakeClock()).search_trip(request)

    assert result.options
    transport_calls = [call for call in gateway.calls if call[0] == "search_transport"]
    assert len(transport_calls) == 4


@pytest.mark.asyncio
async def test_empty_search_never_exceeds_six_transport_calls() -> None:
    request, _, _, _, capabilities = inputs()
    gateway = FakeTutuGateway(capabilities=capabilities)

    result = await TripPlanner(gateway, FakeClock()).search_trip(request)

    assert not result.options
    transport_calls = [call for call in gateway.calls if call[0] == "search_transport"]
    assert len(transport_calls) == 6
    assert all(name != "search_hotels" for name, _ in gateway.calls)


@pytest.mark.asyncio
async def test_hotels_are_searched_for_each_exact_transport_pair_stay() -> None:
    request, overnight, return_offer, hotel, capabilities = inputs()
    daytime = overnight.model_copy(
        update={
            "provider_id": "daytime",
            "departure_at": datetime.fromisoformat("2026-08-21T08:00:00+03:00"),
            "arrival_at": datetime.fromisoformat("2026-08-21T12:00:00+03:00"),
        }
    )
    gateway = FakeTutuGateway(capabilities=capabilities)
    calls = 0

    async def directional_transport(_):
        nonlocal calls
        calls += 1
        return [daytime, overnight] if calls == 1 else [return_offer]

    async def exact_hotels(query):
        gateway.calls.append(("search_hotels", query))
        return [hotel.model_copy(update={"check_in": query.check_in, "check_out": query.check_out})]

    gateway.search_transport = directional_transport
    gateway.search_hotels = exact_hotels

    result = await TripPlanner(gateway, FakeClock()).search_trip(request)

    hotel_queries = [value for name, value in gateway.calls if name == "search_hotels"]
    assert {(query.check_in, query.check_out) for query in hotel_queries} == {
        (date(2026, 8, 21), date(2026, 8, 23)),
        (date(2026, 8, 22), date(2026, 8, 23)),
    }
    assert result.options
    assert all(
        option.combination.hotel.check_in == option.combination.stay.check_in
        for option in result.options
    )


@pytest.mark.asyncio
async def test_budget_failure_is_not_misreported_as_missing_hotel() -> None:
    request, outbound, return_offer, hotel, capabilities = inputs()
    request = request.model_copy(update={"budget": 1})
    gateway = FakeTutuGateway(capabilities=capabilities, hotels=[hotel])
    calls = 0

    async def directional_transport(_):
        nonlocal calls
        calls += 1
        return [outbound] if calls == 1 else [return_offer]

    gateway.search_transport = directional_transport
    result = await TripPlanner(gateway, FakeClock()).search_trip(request)

    assert not result.options
    assert [failure.category for failure in result.failures] == ["over_budget"]


@pytest.mark.asyncio
async def test_transport_stage_is_reused_without_duplicate_provider_calls() -> None:
    request, outbound, return_offer, hotel, capabilities = inputs()
    gateway = FakeTutuGateway(capabilities=capabilities, hotels=[hotel])

    async def directional_transport(query):
        gateway.calls.append(("search_transport", query))
        return [outbound] if query.origin == "Москва" else [return_offer]

    gateway.search_transport = directional_transport
    planner = TripPlanner(gateway, FakeClock())

    candidates = await planner.search_transport_candidates(request)
    transport_only = await planner.complete_transport_candidates(
        candidates,
        search_hotels=False,
    )
    completed = await planner.complete_transport_candidates(candidates, search_hotels=True)

    assert transport_only.options
    assert completed.options
    assert len([item for item in gateway.calls if item[0] == "search_transport"]) == 2
    assert len([item for item in gateway.calls if item[0] == "search_hotels"]) == 1


@pytest.mark.asyncio
async def test_soft_time_windows_are_relaxed_only_to_fill_comparison_options() -> None:
    request, outbound, return_offer, hotel, capabilities = inputs()
    request = request.model_copy(
        update={
            "transport": TransportPreferences(
                departure_window=TimeWindow(from_time=time(18), to_time=time(23)),
                return_window=TimeWindow(from_time=time(17), to_time=time(20)),
            )
        }
    )
    early = outbound.model_copy(
        update={
            "provider_id": "early",
            "departure_at": outbound.departure_at.replace(hour=10),
        }
    )
    noon = outbound.model_copy(
        update={
            "provider_id": "noon",
            "departure_at": outbound.departure_at.replace(hour=12),
        }
    )
    gateway = FakeTutuGateway(capabilities=capabilities, hotels=[hotel])

    async def directional_transport(query):
        return [outbound, early, noon] if query.origin == "Москва" else [return_offer]

    gateway.search_transport = directional_transport

    result = await TripPlanner(gateway, FakeClock()).search_trip(request)

    assert len(result.options) == 3
    relaxed = [
        item
        for item in result.options
        if "Время отправления выходит" in " ".join(item.combination.warnings)
    ]
    assert len(relaxed) == 2
