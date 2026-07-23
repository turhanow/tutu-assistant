import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from app.adapters.mcp_mappers import map_hotel_search, map_transport_search
from app.domain.errors import CheckoutError
from app.domain.models import (
    CheckoutLink,
    HotelStay,
    OfferDetails,
    RankedTripOption,
    SortPreference,
    TransportMode,
    TripCombination,
    TripComponent,
    TripRequest,
    TripSearchResult,
)
from app.services.date_rules import calculate_time_metrics
from app.services.pricing import calculate_price
from app.services.trip_handoff import TripHandoffService

FIXTURES = Path("tests/fixtures/tutu_mcp/fixtures")


class FakeGateway:
    def __init__(self, checkout_url: str = "https://www.tutu.ru/fixture") -> None:
        self.checkout_url = checkout_url
        self.calls = []

    async def get_offer_details(self, ref):
        self.calls.append(("details", ref))
        return OfferDetails(product_type=ref.product_type, title="Подробности")

    async def create_checkout_link(self, ref):
        self.calls.append(("checkout", ref))
        return CheckoutLink(url=self.checkout_url, kind="deeplink")


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def search_result() -> TripSearchResult:
    outbound = map_transport_search(fixture("transport-search.json"))[0]
    return_offer = outbound.model_copy(
        update={
            "provider_id": "return",
            "origin": "Казань",
            "destination": "Москва",
            "departure_at": datetime.fromisoformat("2026-08-23T16:20:00+03:00"),
            "arrival_at": datetime.fromisoformat("2026-08-24T04:50:00+03:00"),
        }
    )
    hotel = map_hotel_search(fixture("hotel-search.json"))[0]
    hotel = hotel.model_copy(update={"check_in": date(2026, 8, 22), "check_out": date(2026, 8, 23)})
    combination = TripCombination(
        outbound=outbound,
        return_offer=return_offer,
        hotel=hotel,
        stay=HotelStay(check_in="2026-08-22", check_out="2026-08-23", nights=1),
        price=calculate_price(outbound, return_offer, hotel),
        metrics=calculate_time_metrics(outbound, return_offer),
        signature="fixture",
    )
    request = TripRequest(
        origin="Москва",
        destination="Казань",
        departure_date="2026-08-21",
        return_date="2026-08-23",
    )
    return TripSearchResult(
        request=request,
        options=(
            RankedTripOption(
                combination=combination,
                labels={SortPreference.BALANCED},
            ),
        ),
        searched_at=datetime(2026, 7, 21, tzinfo=UTC),
    )


def without_direct_urls(result: TripSearchResult) -> TripSearchResult:
    option = result.options[0]
    combination = option.combination
    return result.model_copy(
        update={
            "options": (
                option.model_copy(
                    update={
                        "combination": combination.model_copy(
                            update={
                                "outbound": combination.outbound.model_copy(
                                    update={"provider_url": None}
                                ),
                                "return_offer": combination.return_offer.model_copy(
                                    update={"provider_url": None}
                                ),
                                "hotel": combination.hotel.model_copy(
                                    update={"provider_url": None}
                                ),
                            }
                        )
                    }
                ),
            )
        }
    )


@pytest.mark.asyncio
async def test_details_use_only_local_option_and_component() -> None:
    gateway = FakeGateway()
    service = TripHandoffService(gateway)  # type: ignore[arg-type]

    details = await service.get_details(search_result(), 0, TripComponent.OUTBOUND)

    assert details.title == "Подробности"
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_checkout_returns_separate_tutu_links_for_all_components() -> None:
    gateway = FakeGateway()
    service = TripHandoffService(gateway)  # type: ignore[arg-type]

    items = await service.create_checkout_items(search_result(), 0)

    assert [item.component for item in items] == [
        TripComponent.OUTBOUND,
        TripComponent.RETURN,
        TripComponent.HOTEL,
    ]
    assert not gateway.calls
    assert all(service.is_offer_specific(item) for item in items)


@pytest.mark.asyncio
async def test_checkout_builder_is_used_only_when_offer_has_no_direct_url() -> None:
    gateway = FakeGateway()
    service = TripHandoffService(gateway)  # type: ignore[arg-type]

    items = await service.create_checkout_items(without_direct_urls(search_result()), 0)

    assert len(gateway.calls) == 3
    assert all(service.is_offer_specific(item) for item in items)


@pytest.mark.asyncio
async def test_etrain_is_rejected_when_only_date_level_schedule_link_exists() -> None:
    result = search_result()
    option = result.options[0]
    combination = option.combination
    result = result.model_copy(
        update={
            "options": (
                option.model_copy(
                    update={
                        "combination": combination.model_copy(
                            update={
                                "outbound": combination.outbound.model_copy(
                                    update={"mode": TransportMode.ETRAIN}
                                )
                            }
                        )
                    }
                ),
            )
        }
    )
    service = TripHandoffService(FakeGateway())  # type: ignore[arg-type]

    with pytest.raises(CheckoutError, match="exact checkout links"):
        await service.create_checkout_items(result, 0)


@pytest.mark.asyncio
async def test_checkout_rejects_non_tutu_host() -> None:
    service = TripHandoffService(FakeGateway("https://evil.example/checkout"))  # type: ignore[arg-type]

    with pytest.raises(CheckoutError, match="non-Tutu"):
        await service.create_checkout_items(without_direct_urls(search_result()), 0)


@pytest.mark.asyncio
async def test_stale_local_option_is_rejected_without_provider_call() -> None:
    gateway = FakeGateway()
    service = TripHandoffService(gateway)  # type: ignore[arg-type]

    with pytest.raises(CheckoutError, match="stale"):
        await service.get_details(search_result(), 99, TripComponent.HOTEL)
    assert not gateway.calls
