from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.domain.models import (
    OfferRef,
    PriceStatus,
    PriceSummary,
    SortPreference,
    TransportMode,
    TransportOffer,
    TripCombination,
)
from app.services.date_rules import calculate_time_metrics
from app.services.ranking import calculate_balance_score, rank_options


def test_spec_balance_score_examples() -> None:
    score_a = calculate_balance_score(
        cost_norm=Decimal("0"),
        duration_norm=Decimal(".40"),
        city_time_penalty=Decimal("0"),
        transfers_penalty=Decimal("0"),
        night_penalty=Decimal("0"),
        awkward_time_penalty=Decimal(".25"),
        hotel_quality_penalty=Decimal(".14"),
    )
    score_b = calculate_balance_score(
        cost_norm=Decimal(".50"),
        duration_norm=Decimal("0"),
        city_time_penalty=Decimal("0"),
        transfers_penalty=Decimal(".50"),
        night_penalty=Decimal(".25"),
        awkward_time_penalty=Decimal("0"),
        hotel_quality_penalty=Decimal(".10"),
    )

    assert score_a == Decimal("0.0823")
    assert score_b == Decimal("0.2520")


def test_fastest_minimizes_time_in_transport_not_time_spent_in_city() -> None:
    slow_early = _combination(
        "slow-early",
        outbound_departure=datetime(2026, 8, 21, 8, tzinfo=UTC),
        outbound_hours=10,
        return_departure=datetime(2026, 8, 22, 8, tzinfo=UTC),
        return_hours=10,
        price=Decimal("1000"),
    )
    fast_late = _combination(
        "fast-late",
        outbound_departure=datetime(2026, 8, 21, 9, tzinfo=UTC),
        outbound_hours=5,
        return_departure=datetime(2026, 8, 22, 16, tzinfo=UTC),
        return_hours=5,
        price=Decimal("1200"),
    )

    ranked = rank_options([slow_early, fast_late])
    fastest = next(item for item in ranked if SortPreference.FASTEST in item.labels)

    assert fastest.combination.signature == "fast-late"
    assert fast_late.metrics.total_travel_duration == timedelta(hours=10)
    assert fast_late.metrics.trip_duration > slow_early.metrics.trip_duration


def test_confirmed_sort_preference_controls_first_unique_card() -> None:
    cheap = _combination(
        "cheap",
        outbound_departure=datetime(2026, 8, 21, 8, tzinfo=UTC),
        outbound_hours=10,
        return_departure=datetime(2026, 8, 22, 16, tzinfo=UTC),
        return_hours=10,
        price=Decimal("1000"),
    )
    fast = _combination(
        "fast",
        outbound_departure=datetime(2026, 8, 21, 9, tzinfo=UTC),
        outbound_hours=2,
        return_departure=datetime(2026, 8, 22, 17, tzinfo=UTC),
        return_hours=2,
        price=Decimal("5000"),
    )

    ranked = rank_options([cheap, fast], preferred=SortPreference.FASTEST)

    assert ranked[0].combination.signature == "fast"


def test_balanced_option_values_useful_time_in_city_not_only_short_transport() -> None:
    short_visit = _combination(
        "short-visit",
        outbound_departure=datetime(2026, 8, 21, 14, tzinfo=UTC),
        outbound_hours=2,
        return_departure=datetime(2026, 8, 22, 10, tzinfo=UTC),
        return_hours=2,
        price=Decimal("2000"),
    )
    long_visit = _combination(
        "long-visit",
        outbound_departure=datetime(2026, 8, 21, 8, tzinfo=UTC),
        outbound_hours=2,
        return_departure=datetime(2026, 8, 22, 18, tzinfo=UTC),
        return_hours=2,
        price=Decimal("2000"),
    )

    ranked = rank_options([short_visit, long_visit])
    balanced = next(item for item in ranked if SortPreference.BALANCED in item.labels)

    assert balanced.combination.signature == "long-visit"
    assert long_visit.metrics.time_in_city > short_visit.metrics.time_in_city


def test_ranking_fills_three_distinct_cards_when_one_option_wins_all_labels() -> None:
    combinations = [
        _combination(
            f"option-{index}",
            outbound_departure=datetime(2026, 8, 21, 8 + index, tzinfo=UTC),
            outbound_hours=2 + index,
            return_departure=datetime(2026, 8, 22, 16 + index, tzinfo=UTC),
            return_hours=2 + index,
            price=Decimal(1000 + index * 500),
        )
        for index in range(4)
    ]

    ranked = rank_options(combinations)

    assert len(ranked) == 3
    assert len({item.combination.signature for item in ranked}) == 3
    assert ranked[0].labels == {
        SortPreference.CHEAPEST,
        SortPreference.FASTEST,
        SortPreference.BALANCED,
    }
    assert not ranked[1].labels


def _combination(
    signature: str,
    *,
    outbound_departure: datetime,
    outbound_hours: int,
    return_departure: datetime,
    return_hours: int,
    price: Decimal,
) -> TripCombination:
    outbound = _offer(
        signature + "-out",
        "Москва",
        "Казань",
        outbound_departure,
        outbound_hours,
        price / 2,
    )
    return_offer = _offer(
        signature + "-return",
        "Казань",
        "Москва",
        return_departure,
        return_hours,
        price / 2,
    )
    return TripCombination(
        outbound=outbound,
        return_offer=return_offer,
        price=PriceSummary(
            status=PriceStatus.EXACT,
            known_total=price,
            currency="RUB",
        ),
        metrics=calculate_time_metrics(outbound, return_offer),
        signature=signature,
    )


def _offer(
    identifier: str,
    origin: str,
    destination: str,
    departure: datetime,
    hours: int,
    price: Decimal,
) -> TransportOffer:
    return TransportOffer(
        provider_id=identifier,
        mode=TransportMode.RAIL,
        origin=origin,
        destination=destination,
        departure_at=departure,
        arrival_at=departure + timedelta(hours=hours),
        duration=timedelta(hours=hours),
        price=price,
        currency="RUB",
        transfers=0,
        offer_ref=OfferRef(product_type="rail"),
    )
