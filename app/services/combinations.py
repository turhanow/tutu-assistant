"""Bounded construction of feasible trip combinations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import date, time, timedelta
from decimal import Decimal

from app.domain.errors import TutuAssistantError
from app.domain.models import (
    HotelMode,
    HotelOffer,
    HotelPreferences,
    HotelStay,
    PriceStatus,
    TransportOffer,
    TripCombination,
    TripRequest,
)
from app.services.date_rules import (
    calculate_hotel_stay,
    calculate_time_metrics,
    check_event_feasibility,
)
from app.services.pricing import calculate_price

MAX_TRANSPORT_OFFERS_PER_LEG = 10
MAX_HOTEL_OFFERS = 5
MAX_TRANSPORT_PAIRS = 24
MAX_HOTELS_PER_PAIR = 2
MAX_COMBINATIONS = 50

HotelOffersByStay = Mapping[tuple[date, date], Sequence[HotelOffer]]


def build_combinations(
    request: TripRequest,
    outbound_offers: Sequence[TransportOffer],
    return_offers: Sequence[TransportOffer],
    hotel_offers: Sequence[HotelOffer] = (),
    hotel_offers_by_stay: HotelOffersByStay | None = None,
) -> list[TripCombination]:
    pairs = build_transport_pairs(request, outbound_offers, return_offers)
    combinations: list[TripCombination] = []
    for outbound_offer, return_offer in pairs:
        try:
            stay = calculate_hotel_stay(outbound_offer, return_offer, request.hotel.mode)
        except TutuAssistantError:
            continue
        offers_for_stay = (
            hotel_offers_by_stay.get((stay.check_in, stay.check_out), ())
            if stay is not None and hotel_offers_by_stay is not None
            else hotel_offers
        )
        hotels = _eligible_hotels(offers_for_stay, stay, request.hotel)
        if request.hotel.mode is HotelMode.REQUIRED and not hotels:
            continue
        if request.hotel.mode is not HotelMode.REQUIRED and stay is None:
            combination = _combination(request, outbound_offer, return_offer, None, None)
            if combination is not None:
                combinations.append(combination)
        if request.hotel.mode is not HotelMode.FORBIDDEN and stay is not None:
            for hotel in _select_hotels(hotels):
                combination = _combination(request, outbound_offer, return_offer, hotel, stay)
                if combination is not None:
                    combinations.append(combination)
        if len(combinations) >= MAX_COMBINATIONS:
            break
    return _dedupe_combinations(combinations)[:MAX_COMBINATIONS]


def build_transport_pairs(
    request: TripRequest,
    outbound_offers: Sequence[TransportOffer],
    return_offers: Sequence[TransportOffer],
) -> list[tuple[TransportOffer, TransportOffer]]:
    """Return the bounded set of feasible pairs used by all downstream searches."""
    outbound = _dedupe_transport(outbound_offers)[:MAX_TRANSPORT_OFFERS_PER_LEG]
    returns = _dedupe_transport(return_offers)[:MAX_TRANSPORT_OFFERS_PER_LEG]
    feasible = [
        (outbound_offer, return_offer)
        for outbound_offer in outbound
        for return_offer in returns
        if _feasible_pair(request, outbound_offer, return_offer)
    ]
    return _diverse_transport_pairs(feasible)


def required_hotel_stays(
    request: TripRequest,
    outbound_offers: Sequence[TransportOffer],
    return_offers: Sequence[TransportOffer],
) -> tuple[HotelStay, ...]:
    """Return unique exact stays required by feasible transport pairs."""
    if request.hotel.mode is HotelMode.FORBIDDEN:
        return ()
    unique: dict[tuple[date, date], HotelStay] = {}
    for outbound, return_offer in build_transport_pairs(request, outbound_offers, return_offers):
        try:
            stay = calculate_hotel_stay(outbound, return_offer, request.hotel.mode)
        except TutuAssistantError:
            continue
        if stay is not None:
            unique.setdefault((stay.check_in, stay.check_out), stay)
    return tuple(unique.values())


def _feasible_pair(
    request: TripRequest,
    outbound: TransportOffer,
    return_offer: TransportOffer,
) -> bool:
    if return_offer.departure_at <= outbound.arrival_at:
        return False
    if outbound.departure_at.date() != request.departure_date:
        return False
    if return_offer.departure_at.date() != request.return_date:
        return False
    if request.transport.allowed_modes and (
        outbound.mode not in request.transport.allowed_modes
        or return_offer.mode not in request.transport.allowed_modes
    ):
        return False
    max_transfers = request.transport.max_transfers
    if max_transfers is not None and (
        outbound.transfers is None
        or return_offer.transfers is None
        or outbound.transfers > max_transfers
        or return_offer.transfers > max_transfers
    ):
        return False
    if not _inside_window(outbound.departure_at.time(), request.transport.departure_window):
        return False
    if not _inside_window(return_offer.departure_at.time(), request.transport.return_window):
        return False
    return check_event_feasibility(outbound, return_offer, request.event)


def _inside_window(value: time, window) -> bool:
    if window is None:
        return True
    if window.from_time is not None and value < window.from_time:
        return False
    return window.to_time is None or value <= window.to_time


def _eligible_hotels(
    offers: Sequence[HotelOffer],
    stay: HotelStay | None,
    preferences: HotelPreferences,
) -> list[HotelOffer]:
    if stay is None:
        return []
    eligible: list[HotelOffer] = []
    for hotel in offers[:MAX_HOTEL_OFFERS]:
        if hotel.check_in != stay.check_in or hotel.check_out != stay.check_out:
            continue
        if preferences.stars_min is not None and (
            hotel.stars is None or hotel.stars < preferences.stars_min
        ):
            continue
        if preferences.rating_min is not None and (
            hotel.rating is None or hotel.rating < preferences.rating_min
        ):
            continue
        if preferences.free_cancellation_required and hotel.free_cancellation is not True:
            continue
        if not preferences.required_amenities.issubset(hotel.matched_amenities):
            continue
        if not (preferences.meal_preferences - {"breakfast"}).issubset(hotel.matched_meals):
            continue
        if "breakfast" in preferences.meal_preferences and hotel.breakfast_included is not True:
            continue
        eligible.append(hotel)
    return eligible


def _select_hotels(hotels: Sequence[HotelOffer]) -> list[HotelOffer]:
    if not hotels:
        return []
    cheapest = min(
        hotels,
        key=lambda item: (item.price is None, item.price or Decimal("Infinity")),
    )
    best_rated = max(hotels, key=lambda item: item.rating or Decimal(-1))
    selected = [cheapest]
    if best_rated.offer_id != cheapest.offer_id:
        selected.append(best_rated)
    return selected[:MAX_HOTELS_PER_PAIR]


def _diverse_transport_pairs(
    pairs: Sequence[tuple[TransportOffer, TransportOffer]],
) -> list[tuple[TransportOffer, TransportOffer]]:
    """Keep cheap, fast and convenient pairs instead of truncating a Cartesian product."""

    if not pairs:
        return []
    orderings = (
        sorted(pairs, key=_pair_price_key),
        sorted(pairs, key=_pair_duration_key),
        sorted(pairs, key=_pair_convenience_key),
    )
    selected: list[tuple[TransportOffer, TransportOffer]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for position in range(len(pairs)):
        for ordering in orderings:
            pair = ordering[position]
            signature = _pair_identity(pair)
            if signature in seen:
                continue
            seen.add(signature)
            selected.append(pair)
            if len(selected) == MAX_TRANSPORT_PAIRS:
                return selected
    return selected


def _pair_price_key(pair: tuple[TransportOffer, TransportOffer]) -> tuple:
    prices = (pair[0].price, pair[1].price)
    return (
        any(value is None for value in prices),
        sum((value or Decimal(0) for value in prices), Decimal(0)),
        *_pair_duration_key(pair),
    )


def _pair_duration_key(pair: tuple[TransportOffer, TransportOffer]) -> tuple:
    return (
        sum(
            (offer.duration or (offer.arrival_at - offer.departure_at) for offer in pair),
            start=timedelta(0),
        ),
        pair[0].departure_at,
        pair[1].departure_at,
    )


def _pair_convenience_key(pair: tuple[TransportOffer, TransportOffer]) -> tuple:
    transfers = sum(offer.transfers if offer.transfers is not None else 2 for offer in pair)
    awkward_times = sum(
        int(value.hour < 6) for offer in pair for value in (offer.departure_at, offer.arrival_at)
    )
    return (transfers, awkward_times, *_pair_duration_key(pair), *_pair_price_key(pair)[:2])


def _pair_identity(
    pair: tuple[TransportOffer, TransportOffer],
) -> tuple[str, str, str, str]:
    return (
        pair[0].departure_at.isoformat(),
        pair[0].arrival_at.isoformat(),
        pair[1].departure_at.isoformat(),
        pair[1].arrival_at.isoformat(),
    )


def _combination(
    request: TripRequest,
    outbound: TransportOffer,
    return_offer: TransportOffer,
    hotel: HotelOffer | None,
    stay: HotelStay | None,
) -> TripCombination | None:
    price = calculate_price(outbound, return_offer, hotel)
    if (
        request.budget is not None
        and price.status is PriceStatus.EXACT
        and price.known_total is not None
        and price.known_total > request.budget
    ):
        return None
    signature = _signature(outbound, return_offer, hotel)
    warnings: list[str] = []
    if not outbound.timezone_known or not return_offer.timezone_known:
        warnings.append("Часовой пояс одного из предложений требует проверки на Tutu.")
    if stay is not None and stay.needs_early_checkin_warning:
        warnings.append("Прибытие раньше стандартного времени заселения.")
    if stay is not None and stay.needs_late_checkout_warning:
        warnings.append("Отправление позже стандартного времени выселения.")
    if price.missing_components:
        warnings.append("Итог неполный: не все компоненты имеют подтверждённую цену.")
    return TripCombination(
        outbound=outbound,
        return_offer=return_offer,
        hotel=hotel,
        stay=stay,
        price=price,
        metrics=calculate_time_metrics(outbound, return_offer),
        warnings=tuple(warnings),
        signature=signature,
    )


def _dedupe_transport(offers: Sequence[TransportOffer]) -> list[TransportOffer]:
    unique: dict[str, TransportOffer] = {}
    for offer in offers:
        key = "|".join(
            (
                offer.mode.value,
                offer.departure_at.isoformat(),
                offer.arrival_at.isoformat(),
                str(offer.price),
                offer.currency or "",
            )
        )
        unique.setdefault(key, offer)
    return list(unique.values())


def _dedupe_combinations(
    combinations: Sequence[TripCombination],
) -> list[TripCombination]:
    return list({item.signature: item for item in combinations}.values())


def _signature(
    outbound: TransportOffer,
    return_offer: TransportOffer,
    hotel: HotelOffer | None,
) -> str:
    value = "|".join(
        (
            outbound.provider_id or outbound.departure_at.isoformat(),
            return_offer.provider_id or return_offer.departure_at.isoformat(),
            (hotel.offer_id or hotel.name) if hotel is not None else "no-hotel",
        )
    )
    return hashlib.sha256(value.encode()).hexdigest()[:16]
