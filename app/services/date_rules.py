"""Pure trip date, event feasibility, and time metric rules."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal

from app.domain.errors import TripValidationError, UnsupportedTripError
from app.domain.models import (
    EventConstraint,
    HotelMode,
    HotelStay,
    TransportOffer,
    TripRequest,
    TripTimeMetrics,
)


def validate_planning_date(request: TripRequest, *, today: date) -> None:
    if request.departure_date < today:
        raise TripValidationError("departure date cannot be in the past")
    if (request.return_date - request.departure_date).days > 3:
        raise UnsupportedTripError("P0 supports at most four calendar days")


def calculate_hotel_stay(
    outbound: TransportOffer,
    return_offer: TransportOffer,
    mode: HotelMode,
) -> HotelStay | None:
    if mode is HotelMode.FORBIDDEN:
        return None
    check_in = outbound.arrival_at.date()
    check_out = return_offer.departure_at.date()
    nights = (check_out - check_in).days
    if nights == 0:
        if mode is HotelMode.REQUIRED:
            raise TripValidationError("required hotel needs at least one night")
        return None
    if not 1 <= nights <= 3:
        raise UnsupportedTripError("hotel stay must contain one to three nights")
    return HotelStay(
        check_in=check_in,
        check_out=check_out,
        nights=nights,
        needs_early_checkin_warning=outbound.arrival_at.time() < time(14, 0),
        needs_late_checkout_warning=return_offer.departure_at.time() > time(12, 0),
    )


def check_event_feasibility(
    outbound: TransportOffer,
    return_offer: TransportOffer,
    event: EventConstraint | None,
) -> bool:
    if event is None:
        return True
    destination_tz = outbound.arrival_at.tzinfo
    if event.start_time is not None:
        starts_at = datetime.combine(event.date, event.start_time, tzinfo=destination_tz)
        latest_arrival = starts_at - timedelta(minutes=event.arrival_buffer_minutes)
        if outbound.arrival_at > latest_arrival:
            return False
    if event.end_time is not None:
        end_date = event.end_date or event.date
        ends_at = datetime.combine(end_date, event.end_time, tzinfo=destination_tz)
        earliest_return = ends_at + timedelta(minutes=event.departure_buffer_minutes)
        if return_offer.departure_at < earliest_return:
            return False
    return True


def calculate_time_metrics(
    outbound: TransportOffer,
    return_offer: TransportOffer,
) -> TripTimeMetrics:
    outbound_duration = outbound.duration or (outbound.arrival_at - outbound.departure_at)
    return_duration = return_offer.duration or (return_offer.arrival_at - return_offer.departure_at)
    return TripTimeMetrics(
        total_travel_duration=outbound_duration + return_duration,
        trip_duration=return_offer.arrival_at - outbound.departure_at,
        time_in_city=return_offer.departure_at - outbound.arrival_at,
        night_travel_hours=night_travel_hours(outbound) + night_travel_hours(return_offer),
    )


def night_travel_hours(offer: TransportOffer) -> Decimal:
    current_date = offer.departure_at.date()
    final_date = offer.arrival_at.date()
    total_seconds = 0.0
    while current_date <= final_date:
        start = datetime.combine(current_date, time(0, 0), tzinfo=offer.departure_at.tzinfo)
        end = datetime.combine(current_date, time(6, 0), tzinfo=offer.departure_at.tzinfo)
        overlap_start = max(offer.departure_at, start)
        overlap_end = min(offer.arrival_at, end)
        if overlap_end > overlap_start:
            total_seconds += (overlap_end - overlap_start).total_seconds()
        current_date += timedelta(days=1)
    return Decimal(str(total_seconds / 3600))
