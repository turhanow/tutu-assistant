"""Explainable deterministic ranking across verified destinations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from decimal import Decimal

from app.domain.discovery_models import Recommendation, WeekendProposal
from app.domain.models import Currency, NonNegativeMoney, PriceStatus

SCORE_VERSION = "discovery_v1"


@dataclass(frozen=True, slots=True)
class RankingWeights:
    candidate_match: Decimal = Decimal("0.20")
    confirmed_cost: Decimal = Decimal("0.15")
    travel_duration: Decimal = Decimal("0.10")
    time_in_city: Decimal = Decimal("0.15")
    transfers: Decimal = Decimal("0.08")
    night_travel: Decimal = Decimal("0.07")
    hotel_quality: Decimal = Decimal("0.08")
    content_confidence: Decimal = Decimal("0.10")
    budget_fit: Decimal = Decimal("0.07")

    def __post_init__(self) -> None:
        values = tuple(getattr(self, item.name) for item in fields(self))
        if any(value < 0 for value in values) or sum(values, Decimal(0)) != Decimal(1):
            raise ValueError("ranking weights must be non-negative and sum to one")


DEFAULT_RANKING_WEIGHTS = RankingWeights()


def rank_proposals(
    proposals: Sequence[WeekendProposal],
    *,
    limit: int = 3,
    budget: NonNegativeMoney | None = None,
    budget_currency: Currency | None = None,
    weights: RankingWeights = DEFAULT_RANKING_WEIGHTS,
) -> tuple[Recommendation, ...]:
    if limit < 1 or not proposals:
        return ()
    known_currencies = {
        item.cost.confirmed_currency
        for item in proposals
        if item.cost.confirmed_total is not None and item.cost.confirmed_currency is not None
    }
    comparison_currency = (
        budget_currency
        if budget_currency is not None
        else next(iter(known_currencies))
        if len(known_currencies) == 1
        else None
    )
    costs = [
        item.cost.confirmed_total
        for item in proposals
        if item.cost.confirmed_total is not None
        and item.cost.confirmed_currency == comparison_currency
    ]
    durations = [
        Decimal(str(item.trip_option.combination.metrics.total_travel_duration.total_seconds()))
        for item in proposals
    ]
    city_times = [
        Decimal(str(item.trip_option.combination.metrics.time_in_city.total_seconds()))
        for item in proposals
    ]
    scores = {
        _destination_id(item): _proposal_score(
            item,
            costs=costs,
            durations=durations,
            city_times=city_times,
            budget=budget,
            budget_currency=budget_currency,
            comparison_currency=comparison_currency,
            weights=weights,
        )
        for item in proposals
    }
    ordered = sorted(
        proposals,
        key=lambda item: (-scores[_destination_id(item)], _destination_id(item)),
    )[:limit]
    labels: dict[str, set[str]] = {_destination_id(item): set() for item in ordered}
    labels[_destination_id(ordered[0])].add("best_match")
    cheapest = min(
        (
            item
            for item in ordered
            if item.cost.confirmed_total is not None
            and item.cost.confirmed_currency == comparison_currency
        ),
        key=lambda item: (item.cost.confirmed_total, _destination_id(item)),
        default=None,
    )
    if cheapest is not None:
        labels[_destination_id(cheapest)].add("lowest_confirmed_cost")
    most_time = max(
        ordered,
        key=lambda item: (
            item.trip_option.combination.metrics.time_in_city,
            _reverse_id(_destination_id(item)),
        ),
    )
    labels[_destination_id(most_time)].add("most_time_in_city")
    return tuple(
        Recommendation(
            proposal=item,
            labels=frozenset(labels[_destination_id(item)]),
            score=scores[_destination_id(item)],
            score_version=SCORE_VERSION,
        )
        for item in ordered
    )


def _proposal_score(
    proposal: WeekendProposal,
    *,
    costs: Sequence[Decimal],
    durations: Sequence[Decimal],
    city_times: Sequence[Decimal],
    budget: NonNegativeMoney | None,
    budget_currency: Currency | None,
    comparison_currency: Currency | None,
    weights: RankingWeights,
) -> Decimal:
    cost_score = (
        _lower_is_better(proposal.cost.confirmed_total, costs)
        if proposal.cost.confirmed_total is not None
        and proposal.cost.confirmed_currency == comparison_currency
        else Decimal(0)
    )
    duration = Decimal(
        str(proposal.trip_option.combination.metrics.total_travel_duration.total_seconds())
    )
    city_time = Decimal(str(proposal.trip_option.combination.metrics.time_in_city.total_seconds()))
    combination = proposal.trip_option.combination
    transfer_values = (combination.outbound.transfers, combination.return_offer.transfers)
    transfers_score = (
        Decimal("0.5")
        if any(value is None for value in transfer_values)
        else Decimal(1) - min(Decimal(sum(transfer_values)) / Decimal(4), Decimal(1))
    )
    night_score = Decimal(1) - min(combination.metrics.night_travel_hours / 8, Decimal(1))
    hotel_score = Decimal(1)
    if combination.hotel is not None:
        hotel_score = (
            Decimal("0.5")
            if combination.hotel.rating is None
            else combination.hotel.rating / Decimal(10)
        )
    activity_count = sum(len(day.activities) for day in proposal.days)
    content_score = min(Decimal(activity_count) / Decimal(2), Decimal(1))
    return (
        weights.candidate_match * proposal.candidate.match_score
        + weights.confirmed_cost * cost_score
        + weights.travel_duration * _lower_is_better(duration, durations)
        + weights.time_in_city * _higher_is_better(city_time, city_times)
        + weights.transfers * transfers_score
        + weights.night_travel * night_score
        + weights.hotel_quality * hotel_score
        + weights.content_confidence * content_score
        + weights.budget_fit * _budget_fit(proposal, budget, budget_currency)
    ).quantize(Decimal("0.0001"))


def _budget_fit(
    proposal: WeekendProposal,
    budget: NonNegativeMoney | None,
    budget_currency: Currency | None,
) -> Decimal:
    if budget is None:
        return Decimal("0.5")
    combination = proposal.trip_option.combination
    confirmed = proposal.cost.confirmed_total
    if (
        confirmed is None
        or budget_currency is None
        or proposal.cost.confirmed_currency != budget_currency
    ):
        return Decimal(0)
    if combination.price.status is not PriceStatus.EXACT:
        return Decimal("0.5") if confirmed <= budget else Decimal(0)
    if confirmed <= budget:
        return Decimal(1)
    return max(budget / confirmed, Decimal(0))


def _lower_is_better(value: Decimal, values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal(0)
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        return Decimal(1)
    return (maximum - value) / (maximum - minimum)


def _higher_is_better(value: Decimal, values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal(0)
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        return Decimal(1)
    return (value - minimum) / (maximum - minimum)


def _destination_id(proposal: WeekendProposal) -> str:
    return proposal.candidate.destination.destination_id


def _reverse_id(value: str) -> tuple[int, ...]:
    """Make max() use lexicographically smallest ID for an exact tie."""
    return tuple(-ord(character) for character in value)
