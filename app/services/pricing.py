"""Currency-safe deterministic trip price calculations."""

from __future__ import annotations

from decimal import Decimal

from app.domain.models import HotelOffer, PriceStatus, PriceSummary, TransportOffer


def calculate_price(
    outbound: TransportOffer,
    return_offer: TransportOffer,
    hotel: HotelOffer | None,
) -> PriceSummary:
    components = {
        "outbound": (outbound.price, outbound.currency),
        "return": (return_offer.price, return_offer.currency),
    }
    if hotel is not None:
        components["hotel"] = (hotel.price, hotel.currency)
    known = [(amount, currency) for amount, currency in components.values() if amount is not None]
    missing = tuple(name for name, (amount, _) in components.items() if amount is None)
    if not known:
        return PriceSummary(
            status=PriceStatus.UNKNOWN,
            missing_components=tuple(components),
        )
    currencies = {currency for _, currency in known if currency is not None}
    if len(currencies) != 1:
        return PriceSummary(
            status=PriceStatus.UNKNOWN,
            missing_components=missing,
        )
    total = sum((amount for amount, _ in known), start=Decimal(0))
    status = PriceStatus.EXACT if not missing else PriceStatus.PARTIAL
    return PriceSummary(
        status=status,
        known_total=total,
        currency=currencies.pop(),
        missing_components=missing,
    )
