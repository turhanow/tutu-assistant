"""Deterministic cheapest, fastest, and balanced ranking."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from app.domain.models import (
    PriceStatus,
    RankedTripOption,
    SortPreference,
    TripCombination,
)


def rank_options(
    combinations: Sequence[TripCombination],
    *,
    preferred: SortPreference = SortPreference.BALANCED,
) -> list[RankedTripOption]:
    if not combinations:
        return []
    costs = [
        item.price.known_total
        for item in combinations
        if item.price.status is PriceStatus.EXACT and item.price.known_total is not None
    ]
    durations = [
        Decimal(str(item.metrics.total_travel_duration.total_seconds())) for item in combinations
    ]
    city_times = [Decimal(str(item.metrics.time_in_city.total_seconds())) for item in combinations]
    scores = {
        item.signature: _score(
            item,
            costs=costs,
            durations=durations,
            city_times=city_times,
        )
        for item in combinations
    }
    cheapest = min(
        (
            item
            for item in combinations
            if item.price.status is PriceStatus.EXACT and item.price.known_total is not None
        ),
        key=_cheapest_key,
        default=None,
    )
    fastest = min(combinations, key=_fastest_key)
    balanced = min(combinations, key=lambda item: (scores[item.signature], *_tie_key(item)))
    winners = {
        SortPreference.CHEAPEST: cheapest,
        SortPreference.FASTEST: fastest,
        SortPreference.BALANCED: balanced,
    }
    labels: dict[str, set[SortPreference]] = {}
    for winner, label in (
        (cheapest, SortPreference.CHEAPEST),
        (fastest, SortPreference.FASTEST),
        (balanced, SortPreference.BALANCED),
    ):
        if winner is None:
            continue
        labels.setdefault(winner.signature, set()).add(label)
    ordered: list[TripCombination] = []
    priorities = dict.fromkeys(
        (
            preferred,
            SortPreference.BALANCED,
            SortPreference.CHEAPEST,
            SortPreference.FASTEST,
        )
    )
    for label in priorities:
        winner = winners[label]
        if winner is not None and winner not in ordered:
            ordered.append(winner)
    # Category winners can collapse to one combination. Keep their labels, then fill
    # remaining cards with the next best distinct balanced options so users can compare.
    for candidate in sorted(
        combinations,
        key=lambda item: (scores[item.signature], *_tie_key(item)),
    ):
        if candidate not in ordered:
            ordered.append(candidate)
        if len(ordered) == 3:
            break
    return [
        RankedTripOption(
            combination=item,
            labels=frozenset(labels.get(item.signature, set())),
            balance_score=scores[item.signature],
        )
        for item in ordered[:3]
    ]


def calculate_balance_score(
    *,
    cost_norm: Decimal,
    duration_norm: Decimal,
    city_time_penalty: Decimal,
    transfers_penalty: Decimal,
    night_penalty: Decimal,
    awkward_time_penalty: Decimal,
    hotel_quality_penalty: Decimal,
) -> Decimal:
    return (
        Decimal("0.35") * cost_norm
        + Decimal("0.15") * duration_norm
        + Decimal("0.20") * city_time_penalty
        + Decimal("0.10") * transfers_penalty
        + Decimal("0.08") * night_penalty
        + Decimal("0.05") * awkward_time_penalty
        + Decimal("0.07") * hotel_quality_penalty
    )


def _score(
    item: TripCombination,
    *,
    costs: Sequence[Decimal],
    durations: Sequence[Decimal],
    city_times: Sequence[Decimal],
) -> Decimal:
    exact_cost = item.price.known_total if item.price.status is PriceStatus.EXACT else None
    cost_norm = _normalize(exact_cost, costs) if exact_cost is not None else Decimal(1)
    duration = Decimal(str(item.metrics.total_travel_duration.total_seconds()))
    city_time = Decimal(str(item.metrics.time_in_city.total_seconds()))
    transfers = item.outbound.transfers
    return_transfers = item.return_offer.transfers
    transfers_penalty = (
        Decimal("0.5")
        if transfers is None or return_transfers is None
        else min(Decimal(transfers + return_transfers) / 2, Decimal(1))
    )
    night_penalty = min(item.metrics.night_travel_hours / 8, Decimal(1))
    awkward_flags = (
        _awkward(item.outbound.departure_at.hour),
        _awkward(item.outbound.arrival_at.hour),
        _awkward(item.return_offer.departure_at.hour),
        _awkward(item.return_offer.arrival_at.hour),
    )
    awkward = Decimal(sum(awkward_flags)) / 4
    hotel_penalty = Decimal(0)
    if item.hotel is not None:
        hotel_penalty = (
            Decimal("0.5")
            if item.hotel.rating is None
            else max(min((Decimal(10) - item.hotel.rating) / 10, Decimal(1)), Decimal(0))
        )
    return calculate_balance_score(
        cost_norm=cost_norm,
        duration_norm=_normalize(duration, durations),
        city_time_penalty=Decimal(1) - _normalize(city_time, city_times),
        transfers_penalty=transfers_penalty,
        night_penalty=night_penalty,
        awkward_time_penalty=awkward,
        hotel_quality_penalty=hotel_penalty,
    )


def _normalize(value: Decimal, values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal(1)
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return Decimal(0)
    return (value - minimum) / (maximum - minimum)


def _awkward(hour: int) -> int:
    return int(0 <= hour < 6)


def _unknown_count(item: TripCombination) -> int:
    return sum(
        value is None
        for value in (
            item.outbound.price,
            item.return_offer.price,
            item.outbound.transfers,
            item.return_offer.transfers,
            item.hotel.rating if item.hotel else Decimal(0),
        )
    )


def _tie_key(item: TripCombination) -> tuple:
    return (
        _unknown_count(item),
        item.price.known_total or Decimal("Infinity"),
        item.metrics.total_travel_duration,
        (item.outbound.transfers or 0) + (item.return_offer.transfers or 0),
        item.signature,
    )


def _cheapest_key(item: TripCombination) -> tuple:
    return (item.price.known_total, *_tie_key(item))


def _fastest_key(item: TripCombination) -> tuple:
    return (item.metrics.total_travel_duration, *_tie_key(item))
