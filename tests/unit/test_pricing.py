from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.domain.models import (
    HotelOffer,
    OfferRef,
    PriceStatus,
    TransportMode,
    TransportOffer,
)
from app.services.pricing import calculate_price


def transport(price: str | None, currency: str | None = "RUB") -> TransportOffer:
    departure = datetime(2026, 8, 21, 8, 0, tzinfo=UTC)
    return TransportOffer(
        provider_id="fixture",
        mode=TransportMode.RAIL,
        origin="Москва",
        destination="Казань",
        departure_at=departure,
        arrival_at=departure + timedelta(hours=8),
        price=price,
        currency=currency if price is not None else None,
        offer_ref=OfferRef(product_type="rail"),
    )


def hotel(price: str | None, currency: str | None = "RUB") -> HotelOffer:
    return HotelOffer(
        offer_id="fixture",
        name="Hotel",
        check_in=date(2026, 8, 21),
        check_out=date(2026, 8, 23),
        price=price,
        currency=currency if price is not None else None,
        offer_ref=OfferRef(product_type="hotels"),
    )


def test_exact_price_uses_decimal_and_whole_stay_hotel_total() -> None:
    result = calculate_price(transport("4500"), transport("4700"), hotel("9000"))
    assert result.status is PriceStatus.EXACT
    assert result.known_total == Decimal("18200")


def test_missing_component_is_partial() -> None:
    result = calculate_price(transport("4500"), transport(None), hotel("9000"))
    assert result.status is PriceStatus.PARTIAL
    assert result.known_total == Decimal("13500")
    assert result.missing_components == ("return",)


def test_mixed_currencies_are_unknown_and_never_summed() -> None:
    result = calculate_price(transport("4500", "RUB"), transport("100", "USD"), None)
    assert result.status is PriceStatus.UNKNOWN
    assert result.known_total is None
