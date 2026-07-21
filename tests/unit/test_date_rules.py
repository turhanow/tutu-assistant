from datetime import UTC, date, datetime, timedelta

import pytest

from app.domain.errors import TripValidationError
from app.domain.models import (
    EventConstraint,
    HotelMode,
    OfferRef,
    TransportMode,
    TransportOffer,
    TripRequest,
)
from app.services.date_rules import (
    calculate_hotel_stay,
    calculate_time_metrics,
    check_event_feasibility,
    validate_planning_date,
)


def offer(departure: datetime, arrival: datetime) -> TransportOffer:
    return TransportOffer(
        provider_id="fixture",
        mode=TransportMode.RAIL,
        origin="Москва",
        destination="Казань",
        departure_at=departure,
        arrival_at=arrival,
        price="1000",
        currency="RUB",
        offer_ref=OfferRef(product_type="rail"),
    )


def test_past_date_is_validated_with_explicit_today() -> None:
    request = TripRequest(
        origin="Москва",
        destination="Казань",
        departure_date=date(2026, 8, 20),
        return_date=date(2026, 8, 21),
    )
    with pytest.raises(TripValidationError, match="past"):
        validate_planning_date(request, today=date(2026, 8, 21))


def test_stay_uses_actual_arrival_and_return_dates() -> None:
    outbound = offer(
        datetime(2026, 8, 20, 23, 0, tzinfo=UTC),
        datetime(2026, 8, 21, 8, 0, tzinfo=UTC),
    )
    return_offer = offer(
        datetime(2026, 8, 23, 18, 0, tzinfo=UTC),
        datetime(2026, 8, 24, 2, 0, tzinfo=UTC),
    )
    stay = calculate_hotel_stay(outbound, return_offer, HotelMode.REQUIRED)

    assert stay is not None
    assert stay.nights == 2
    assert stay.needs_early_checkin_warning is True
    assert stay.needs_late_checkout_warning is True


def test_same_day_required_hotel_is_rejected() -> None:
    outbound = offer(
        datetime(2026, 8, 21, 7, 0, tzinfo=UTC),
        datetime(2026, 8, 21, 10, 0, tzinfo=UTC),
    )
    return_offer = offer(
        datetime(2026, 8, 21, 21, 0, tzinfo=UTC),
        datetime(2026, 8, 22, 0, 30, tzinfo=UTC),
    )
    with pytest.raises(TripValidationError):
        calculate_hotel_stay(outbound, return_offer, HotelMode.REQUIRED)
    assert calculate_hotel_stay(outbound, return_offer, HotelMode.FORBIDDEN) is None


def test_event_buffers_filter_late_arrival_and_early_return() -> None:
    outbound = offer(
        datetime(2026, 8, 21, 12, 0, tzinfo=UTC),
        datetime(2026, 8, 21, 17, 31, tzinfo=UTC),
    )
    return_offer = offer(
        datetime(2026, 8, 21, 23, 29, tzinfo=UTC),
        datetime(2026, 8, 22, 4, 0, tzinfo=UTC),
    )
    event = EventConstraint(
        date=date(2026, 8, 21),
        start_time="19:00",
        end_time="22:30",
    )
    assert check_event_feasibility(outbound, return_offer, event) is False


def test_time_metrics_include_night_travel() -> None:
    outbound = offer(
        datetime(2026, 8, 21, 23, 0, tzinfo=UTC),
        datetime(2026, 8, 22, 5, 0, tzinfo=UTC),
    )
    return_offer = offer(
        datetime(2026, 8, 23, 20, 0, tzinfo=UTC),
        datetime(2026, 8, 24, 2, 0, tzinfo=UTC),
    )
    metrics = calculate_time_metrics(outbound, return_offer)
    assert metrics.trip_duration == timedelta(hours=51)
    assert metrics.time_in_city == timedelta(hours=39)
    assert metrics.night_travel_hours == 7
