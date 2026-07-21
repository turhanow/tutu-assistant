"""Keep verified and estimated proposal costs explicitly separated."""

from __future__ import annotations

from decimal import Decimal

from app.domain.content_models import DestinationContent, PriceRange, sum_price_ranges
from app.domain.discovery_models import CostBreakdown, DayPlan
from app.domain.models import RankedTripOption


class TripBudgetBuilder:
    def build(
        self,
        trip_option: RankedTripOption,
        content: DestinationContent,
        days: tuple[DayPlan, ...],
    ) -> CostBreakdown:
        price = trip_option.combination.price
        confirmed_total = price.known_total
        confirmed_currency = price.currency if confirmed_total is not None else None
        unknown = list(price.missing_components)

        estimate = self._local_estimate(content, days)
        if (
            estimate is not None
            and confirmed_currency is not None
            and estimate.currency != confirmed_currency
        ):
            estimate = None
            unknown.append("local_expenses_other_currency")

        scheduled = tuple(
            scheduled_activity.activity for day in days for scheduled_activity in day.activities
        )
        if estimate is None:
            unknown.extend(("food", "local_transport"))
            if any(activity.estimated_cost is None for activity in scheduled) or not scheduled:
                unknown.append("activities")

        return CostBreakdown(
            confirmed_total=confirmed_total,
            confirmed_currency=confirmed_currency,
            estimated=estimate,
            unknown_components=tuple(dict.fromkeys(unknown)),
        )

    @staticmethod
    def _local_estimate(
        content: DestinationContent,
        days: tuple[DayPlan, ...],
    ) -> PriceRange | None:
        active_days = sum(bool(day.activities) for day in days)
        if content.destination.estimated_daily_cost is not None and active_days:
            daily = content.destination.estimated_daily_cost
            multiplier = Decimal(active_days)
            return PriceRange(
                minimum=daily.minimum * multiplier,
                maximum=daily.maximum * multiplier,
                currency=daily.currency,
            )
        estimates = tuple(
            item.activity.estimated_cost
            for day in days
            for item in day.activities
            if item.activity.estimated_cost is not None
        )
        return sum_price_ranges(estimates)
