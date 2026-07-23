from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.domain.models import (
    EventConstraint,
    HotelMode,
    HotelPreferences,
    HotelStay,
    OfferRef,
    TransportMode,
    TransportOffer,
    TravelerComposition,
    TripRequest,
)


def trip_request(**overrides) -> TripRequest:
    values = {
        "origin": "Москва",
        "destination": "Казань",
        "departure_date": date(2026, 8, 21),
        "return_date": date(2026, 8, 23),
    }
    values.update(overrides)
    return TripRequest(**values)


def test_minimal_trip_request_has_p0_defaults() -> None:
    request = trip_request(budget="25000.50")

    assert request.travelers == TravelerComposition(adults=1, children=0, rooms=1)
    assert request.hotel.mode is HotelMode.REQUIRED
    assert request.budget == Decimal("25000.50")
    assert request.currency == "RUB"


def test_overnight_trip_normalizes_optional_hotel_and_refusal_to_day_trip() -> None:
    optional = trip_request(hotel=HotelPreferences(mode=HotelMode.OPTIONAL))

    assert optional.hotel.mode is HotelMode.REQUIRED
    without_hotel = trip_request(hotel=HotelPreferences(mode=HotelMode.FORBIDDEN))
    assert without_hotel.return_date == without_hotel.departure_date
    assert without_hotel.hotel.mode is HotelMode.FORBIDDEN


@pytest.mark.parametrize(
    ("field", "value"),
    [("adults", 0), ("adults", 3), ("children", 1), ("rooms", 2)],
)
def test_traveler_composition_enforces_p0_scope(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        TravelerComposition(**{field: value})


def test_trip_request_rejects_same_route_and_invalid_date_range() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        trip_request(destination="москва")

    with pytest.raises(ValidationError, match="cannot precede"):
        trip_request(return_date=date(2026, 8, 20))

    with pytest.raises(ValidationError, match="four calendar days"):
        trip_request(return_date=date(2026, 8, 25))


def test_event_must_fit_inside_trip() -> None:
    with pytest.raises(ValidationError, match="inside the trip"):
        trip_request(event=EventConstraint(date=date(2026, 8, 24)))


def test_hotel_stay_nights_must_match_dates() -> None:
    stay = HotelStay(
        check_in=date(2026, 8, 21),
        check_out=date(2026, 8, 23),
        nights=2,
    )
    assert stay.nights == 2

    with pytest.raises(ValidationError, match="must match"):
        HotelStay(
            check_in=date(2026, 8, 21),
            check_out=date(2026, 8, 23),
            nights=1,
        )


def test_transport_offer_requires_checkout_reference_and_https_url() -> None:
    departure = datetime(2026, 8, 21, 8, 0, tzinfo=UTC)
    offer = TransportOffer(
        provider_id="fixture",
        mode=TransportMode.RAIL,
        origin="Москва",
        destination="Казань",
        departure_at=departure,
        arrival_at=departure + timedelta(hours=12),
        price="4500.00",
        currency="RUB",
        provider_url="https://www.tutu.ru/fixture",
        offer_ref=OfferRef(product_type="rail"),
    )

    assert offer.price == Decimal("4500.00")
    assert offer.duration is None

    with pytest.raises(ValidationError, match="HTTPS"):
        offer.model_copy(update={"provider_url": "http://example.com"}, deep=True).model_validate(
            {
                **offer.model_dump(),
                "provider_url": "http://example.com",
            }
        )


def test_hotel_filters_are_immutable() -> None:
    preferences = HotelPreferences(
        mode=HotelMode.REQUIRED,
        meal_preferences={"breakfast"},
    )
    assert preferences.meal_preferences == frozenset({"breakfast"})
