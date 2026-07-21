"""Strict mapping from untrusted Tutu MCP payloads into domain models."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from app.domain.errors import ProviderResponseError
from app.domain.models import (
    CheckoutLink,
    HotelOffer,
    HotelRateDetails,
    HotelRoomDetails,
    OfferDetails,
    OfferRef,
    TransportMode,
    TransportOffer,
)

TRANSPORT_MODES = {
    "avia": TransportMode.AVIA,
    "rail": TransportMode.RAIL,
    "railway": TransportMode.RAIL,
    "bus": TransportMode.BUS,
    "etrain": TransportMode.ETRAIN,
    "mixed": TransportMode.MIXED,
}


def decimal_value(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ProviderResponseError("provider returned an invalid decimal value") from error


def parse_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ProviderResponseError(f"provider field {field} must be an ISO datetime")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ProviderResponseError(f"provider field {field} is not an ISO datetime") from error


def parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ProviderResponseError(f"provider field {field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ProviderResponseError(f"provider field {field} is not an ISO date") from error


def money(payload: Any) -> tuple[Decimal | None, str | None]:
    if payload is None:
        return None, None
    if not isinstance(payload, dict):
        raise ProviderResponseError("provider price must be an object")
    amount = decimal_value(payload.get("amount"))
    currency = payload.get("currency")
    if currency is not None and not isinstance(currency, str):
        raise ProviderResponseError("provider currency must be a string")
    return amount, currency


def map_transport_search(payload: dict[str, Any]) -> list[TransportOffer]:
    variants = payload.get("variants", payload.get("offers"))
    meta = payload.get("meta")
    if not isinstance(variants, list) or not isinstance(meta, dict):
        raise ProviderResponseError("transport response must contain variants and meta")
    origin = _location_name(meta.get("from"), "from")
    destination = _location_name(meta.get("to"), "to")
    offers: list[TransportOffer] = []
    for row in variants:
        if not isinstance(row, dict):
            raise ProviderResponseError("transport variant must be an object")
        price, currency = money(row.get("price"))
        mode = TRANSPORT_MODES.get(str(row.get("transport")), TransportMode.UNKNOWN)
        checkout_ref = row.get("checkout_ref")
        if not isinstance(checkout_ref, dict):
            raise ProviderResponseError("transport variant is missing checkout_ref")
        try:
            offers.append(
                TransportOffer(
                    provider_id=_optional_string(row.get("offer_id")),
                    mode=mode,
                    origin=origin,
                    destination=destination,
                    departure_at=parse_datetime(row.get("departure_at"), "departure_at"),
                    arrival_at=parse_datetime(row.get("arrival_at"), "arrival_at"),
                    duration=_duration(row.get("duration_min")),
                    price=price,
                    currency=currency,
                    transfers=_transfers(row),
                    provider_url=row.get("checkout_url") or row.get("search_results_url"),
                    offer_ref=OfferRef(
                        product_type=str(checkout_ref.get("transport") or row.get("transport")),
                        checkout_data=checkout_ref,
                        details_data=row.get("details_ref") or {},
                    ),
                    timezone_known=_timezone_known(row),
                )
            )
        except ValidationError as error:
            raise ProviderResponseError("invalid transport offer returned by provider") from error
    return offers


def map_hotel_search(payload: dict[str, Any]) -> list[HotelOffer]:
    hotels = payload.get("hotels")
    stay = payload.get("stay")
    if not isinstance(hotels, list) or not isinstance(stay, dict):
        raise ProviderResponseError("hotel response must contain hotels and stay")
    check_in = parse_date(stay.get("check_in"), "stay.check_in")
    check_out = parse_date(stay.get("check_out"), "stay.check_out")
    offers: list[HotelOffer] = []
    for row in hotels:
        if not isinstance(row, dict):
            raise ProviderResponseError("hotel row must be an object")
        best = row.get("best_offer")
        checkout_ref = row.get("checkout_ref")
        if not isinstance(best, dict) or not isinstance(checkout_ref, dict):
            raise ProviderResponseError("hotel row is missing best_offer or checkout_ref")
        price, currency = money(best.get("price"))
        try:
            offers.append(
                HotelOffer(
                    hotel_id=_optional_string(row.get("hotel_id")),
                    offer_id=_optional_string(row.get("tutu_offer_id")),
                    name=row.get("name"),
                    check_in=check_in,
                    check_out=check_out,
                    price=price,
                    currency=currency,
                    stars=row.get("stars"),
                    rating=decimal_value(row.get("rating")),
                    review_count=row.get("review_count"),
                    room_name=best.get("room_name"),
                    breakfast_included=best.get("breakfast_included"),
                    free_cancellation=best.get("free_cancellation"),
                    provider_url=best.get("checkout_url") or row.get("checkout_url"),
                    offer_ref=OfferRef(
                        product_type="hotels",
                        checkout_data=checkout_ref,
                        details_data={
                            "offer_id": row.get("hotel_id"),
                            "check_in": check_in.isoformat(),
                            "check_out": check_out.isoformat(),
                            "adults": checkout_ref.get("adults", 1),
                            "view": "compact",
                        },
                    ),
                )
            )
        except ValidationError as error:
            raise ProviderResponseError("invalid hotel offer returned by provider") from error
    return offers


def map_checkout_link(payload: dict[str, Any]) -> CheckoutLink:
    try:
        return CheckoutLink(
            url=payload.get("checkout_url"),
            fallback_url=payload.get("fallback_url"),
        )
    except ValidationError as error:
        raise ProviderResponseError("provider returned an invalid checkout link") from error


def map_offer_details(payload: dict[str, Any], product_type: str) -> OfferDetails:
    if product_type != "hotels":
        title = payload.get("title") or payload.get("name") or f"{product_type} offer"
        return OfferDetails(
            product_type=product_type,
            title=str(title),
            amenities=_transport_amenities(payload),
            facts=_transport_facts(payload),
        )
    hotel = payload.get("hotel")
    rooms = payload.get("rooms")
    if not isinstance(hotel, dict) or not isinstance(rooms, list):
        raise ProviderResponseError("hotel details must contain hotel and rooms")
    try:
        return OfferDetails(
            product_type="hotels",
            title=hotel.get("name"),
            check_in_time=_optional_time(hotel.get("check_in_time")),
            check_out_time=_optional_time(hotel.get("check_out_time")),
            amenities=_hotel_amenities(hotel.get("amenity_groups")),
            rooms=tuple(_hotel_room(row) for row in rooms),
        )
    except (ValidationError, TypeError) as error:
        raise ProviderResponseError("invalid hotel details returned by provider") from error


def _location_name(value: Any, field: str) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        raise ProviderResponseError(f"transport meta.{field}.name is missing")
    return value["name"]


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _duration(value: Any) -> timedelta | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or value < 0:
        raise ProviderResponseError("duration_min must be non-negative")
    return timedelta(minutes=value)


def _transfers(row: dict[str, Any]) -> int | None:
    segments = row.get("segments_count")
    legs = row.get("legs")
    if not isinstance(segments, int) or not isinstance(legs, list):
        return None
    return max(segments - len(legs), 0)


def _timezone_known(row: dict[str, Any]) -> bool:
    return all(
        parse_datetime(row.get(field), field).tzinfo is not None
        for field in ("departure_at", "arrival_at")
    )


def _optional_time(value: Any) -> time | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProviderResponseError("hotel check-in/out time must be an ISO time")
    try:
        return time.fromisoformat(value)
    except ValueError as error:
        raise ProviderResponseError("hotel check-in/out time is invalid") from error


def _hotel_amenities(groups: Any) -> tuple[str, ...]:
    if not isinstance(groups, list):
        return ()
    values: list[str] = []
    for group in groups:
        if isinstance(group, dict) and isinstance(group.get("amenities"), list):
            values.extend(str(item) for item in group["amenities"])
    return tuple(dict.fromkeys(values))


def _hotel_room(row: Any) -> HotelRoomDetails:
    if not isinstance(row, dict) or not isinstance(row.get("rates"), list):
        raise ProviderResponseError("hotel room must contain rates")
    rates: list[HotelRateDetails] = []
    for rate in row["rates"]:
        if not isinstance(rate, dict):
            raise ProviderResponseError("hotel rate must be an object")
        price, currency = money(rate.get("price"))
        rates.append(
            HotelRateDetails(
                price=price,
                currency=currency,
                breakfast_included=rate.get("breakfast_included"),
                free_cancellation=rate.get("free_cancellation"),
                refundable=rate.get("refundable"),
                pay_online=rate.get("pay_online"),
                checkout_url=rate.get("checkout_url"),
                offer_pack_ref=_optional_string(rate.get("offerpack_hash")),
            )
        )
    return HotelRoomDetails(
        name=row.get("room_name"),
        bed_summary=row.get("bed_summary"),
        max_occupancy=row.get("max_occupancy"),
        rates=tuple(rates),
    )


def _transport_facts(payload: dict[str, Any]) -> tuple[str, ...]:
    facts: list[str] = []
    carriers = payload.get("carriers")
    if isinstance(carriers, list) and carriers:
        facts.append("Перевозчик: " + ", ".join(str(item) for item in carriers[:3]))
    duration = payload.get("duration_min")
    if isinstance(duration, int | float) and duration >= 0:
        hours, minutes = divmod(int(duration), 60)
        facts.append(f"Время в пути: {hours} ч {minutes} мин")
    review = payload.get("review_summary")
    if isinstance(review, dict) and isinstance(review.get("label"), str):
        facts.append(str(review["label"]))
    ticket = payload.get("ticket")
    if isinstance(ticket, dict) and ticket.get("electronic") is True:
        facts.append("Электронный билет")
    return tuple(facts)


def _transport_amenities(payload: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    raw = payload.get("amenities")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                label = item.get("label") or item.get("name") or item.get("code")
                if label:
                    values.append(str(label))
            elif isinstance(item, str):
                values.append(item)
    return tuple(dict.fromkeys(values))
