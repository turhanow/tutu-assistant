import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from app.adapters.mcp_mappers import (
    decimal_value,
    map_checkout_link,
    map_hotel_search,
    map_offer_details,
    map_transport_search,
    money,
    parse_date,
    parse_datetime,
)
from app.domain.errors import ProviderResponseError
from app.domain.models import TransportMode

FIXTURES = Path("tests/fixtures/tutu_mcp/fixtures")


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_maps_live_transport_fixture_to_domain() -> None:
    offers = map_transport_search(fixture("transport-search.json"))

    assert len(offers) == 1
    offer = offers[0]
    assert offer.mode is TransportMode.RAIL
    assert offer.origin == "Москва"
    assert offer.destination == "Казань"
    assert offer.price == Decimal("2587.61")
    assert offer.duration == timedelta(minutes=754)
    assert offer.transfers == 0
    assert offer.service_number == "128М"
    assert offer.carrier == "ФПК"
    assert offer.timezone_known is True


@pytest.mark.parametrize("name", ["rail-search.json", "avia-search.json"])
def test_maps_targeted_transport_fixtures(name: str) -> None:
    offers = map_transport_search(fixture(name))
    assert offers
    assert offers[0].price is not None


def test_maps_live_hotel_fixture_as_whole_stay_price() -> None:
    offers = map_hotel_search(fixture("hotel-search.json"))

    assert len(offers) == 1
    offer = offers[0]
    assert offer.price == Decimal("16250.0")
    assert offer.currency == "RUB"
    assert offer.room_name == "Апартаменты L"
    assert offer.free_cancellation is True


def test_maps_live_checkout_fixture() -> None:
    checkout = map_checkout_link(fixture("checkout-link.json"))
    assert str(checkout.url) == "https://www.tutu.ru/fixture"
    assert checkout.kind == "deeplink"


def test_maps_live_hotel_details_without_raw_payload() -> None:
    details = map_offer_details(fixture("hotel-details.json"), "hotels")

    assert details.title == 'Апарт Отель "Центральный"'
    assert details.check_in_time.isoformat() == "15:00:00"
    assert len(details.rooms) == 4
    assert details.rooms[0].rates[0].price == Decimal("15340.0")
    assert "Бесплатный Wi-Fi" in details.amenities


def test_rejects_malformed_provider_response() -> None:
    with pytest.raises(ProviderResponseError):
        map_transport_search({"variants": [{}], "meta": {}})

    with pytest.raises(ProviderResponseError):
        map_hotel_search({"hotels": [{}], "stay": {}})


def test_maps_grounded_transport_details_without_preserving_raw_payload() -> None:
    details = map_offer_details(
        {
            "title": "Поезд 128М",
            "carriers": ["ФПК"],
            "duration_min": 754,
            "review_summary": {"label": "⭐ 7.3/10 · 64 отзыва"},
            "ticket": {"electronic": True},
            "amenities": [
                {"label": "Wi-Fi"},
                {"name": "Кондиционер"},
                "Розетка",
            ],
        },
        "railway",
    )

    assert details.title == "Поезд 128М"
    assert "Перевозчик: ФПК" in details.facts
    assert "Время в пути: 12 ч 34 мин" in details.facts
    assert "Электронный билет" in details.facts
    assert details.amenities == ("Wi-Fi", "Кондиционер", "Розетка")


@pytest.mark.parametrize(
    "operation",
    [
        lambda: decimal_value("not-money"),
        lambda: parse_datetime(123, "departure_at"),
        lambda: parse_datetime("not-a-date", "departure_at"),
        lambda: parse_date(123, "check_in"),
        lambda: parse_date("2026-99-99", "check_in"),
        lambda: money("100 RUB"),
        lambda: money({"amount": 1, "currency": 123}),
        lambda: map_transport_search({"variants": "bad", "meta": {}}),
        lambda: map_transport_search({"variants": ["bad"], "meta": {"from": {}, "to": {}}}),
        lambda: map_hotel_search({"hotels": "bad", "stay": {}}),
        lambda: map_offer_details({"hotel": {}, "rooms": "bad"}, "hotels"),
        lambda: map_checkout_link({"checkout_url": "http://unsafe.example"}),
    ],
)
def test_boundary_rejects_each_malformed_primitive(operation) -> None:
    with pytest.raises(ProviderResponseError):
        operation()
