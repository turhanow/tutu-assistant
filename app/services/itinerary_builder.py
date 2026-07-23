"""Deterministic activity scheduling inside the verified time in a destination."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from app.domain.content_models import Activity, DestinationContent
from app.domain.discovery_models import DayPlan, ScheduledActivity
from app.domain.models import RankedTripOption


@dataclass(frozen=True, slots=True)
class ItineraryBuildResult:
    days: tuple[DayPlan, ...]
    unscheduled_activity_ids: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def activity_count(self) -> int:
        return sum(len(day.activities) + len(day.suggestions) for day in self.days)

    @property
    def content_complete(self) -> bool:
        return bool(self.days) and all(
            2 <= len(day.activities) + len(day.suggestions) <= 3 for day in self.days
        )


class ItineraryBuilder:
    """Build a conservative plan without inventing opening hours."""

    def __init__(
        self,
        *,
        day_start: time = time(10),
        day_end: time = time(19),
        arrival_buffer: timedelta = timedelta(hours=1),
        return_buffer: timedelta = timedelta(minutes=90),
        activity_gap: timedelta = timedelta(minutes=30),
        max_days: int = 4,
    ) -> None:
        if day_end <= day_start:
            raise ValueError("planning day must end after it starts")
        if min(arrival_buffer, return_buffer, activity_gap) < timedelta(0):
            raise ValueError("planning buffers cannot be negative")
        if not 1 <= max_days <= 4:
            raise ValueError("max_days must be between one and four")
        self._day_start = day_start
        self._day_end = day_end
        self._arrival_buffer = arrival_buffer
        self._return_buffer = return_buffer
        self._activity_gap = activity_gap
        self._max_days = max_days

    def build(
        self,
        content: DestinationContent,
        trip_option: RankedTripOption,
        *,
        verified_at: datetime,
        preferred_categories: Iterable[str] = (),
    ) -> ItineraryBuildResult:
        combination = trip_option.combination
        arrival = combination.outbound.arrival_at
        departure = combination.return_offer.departure_at
        activities, invalid_ids = self._eligible_activities(content, verified_at)
        preferred = frozenset(item.casefold() for item in preferred_categories)
        original_order = {item.activity_id: index for index, item in enumerate(activities)}
        activities.sort(
            key=lambda item: (
                not bool(item.categories.intersection(preferred)),
                original_order[item.activity_id],
                item.activity_id,
            )
        )

        if (
            not combination.outbound.timezone_known
            or not combination.return_offer.timezone_known
            or arrival.tzinfo is None
            or departure.tzinfo is None
        ):
            dates = tuple(self._dates(arrival.date(), departure.date()))[: self._max_days]
            allocations: list[list[Activity]] = [[] for _ in dates]
            for index, activity in enumerate(activities[: len(dates) * 3]):
                allocations[index % len(dates)].append(activity)
            allocated_ids = {
                activity.activity_id for allocation in allocations for activity in allocation
            }
            return ItineraryBuildResult(
                days=tuple(
                    DayPlan(date=plan_date, suggestions=tuple(allocation))
                    for plan_date, allocation in zip(dates, allocations, strict=True)
                ),
                unscheduled_activity_ids=tuple(
                    item.activity_id
                    for item in activities
                    if item.activity_id not in allocated_ids
                ),
                warnings=("Часовой пояс расписания не подтверждён",),
            )

        available_from = arrival + self._arrival_buffer
        available_until = departure - self._return_buffer
        all_dates = tuple(self._dates(arrival.date(), departure.date()))
        dates = all_dates[: self._max_days]
        remaining = list(activities)
        allocations: list[list[Activity]] = [[] for _ in dates]
        if dates:
            for index, activity in enumerate(remaining[: len(dates) * 3]):
                allocations[index % len(dates)].append(activity)
        allocated_ids = {
            activity.activity_id for allocation in allocations for activity in allocation
        }
        remaining = [item for item in remaining if item.activity_id not in allocated_ids]
        days: list[DayPlan] = []
        untimed: list[Activity] = []
        for plan_date, allocation in zip(dates, allocations, strict=True):
            local_start = datetime.combine(plan_date, self._day_start, tzinfo=arrival.tzinfo)
            local_end = datetime.combine(plan_date, self._day_end, tzinfo=arrival.tzinfo)
            slot_start = max(local_start, available_from)
            slot_end = min(local_end, available_until)
            scheduled: list[ScheduledActivity] = []
            suggestions: list[Activity] = []
            cursor = slot_start
            for item in allocation:
                activity_end = cursor + item.duration
                if cursor < slot_end and activity_end <= slot_end:
                    scheduled.append(
                        ScheduledActivity(
                            activity=item,
                            starts_at=cursor,
                            ends_at=activity_end,
                        )
                    )
                    cursor = activity_end + self._activity_gap
                else:
                    suggestions.append(item)
                    untimed.append(item)
            days.append(
                DayPlan(
                    date=plan_date,
                    activities=tuple(scheduled),
                    suggestions=tuple(suggestions),
                )
            )

        if not days:
            days.append(DayPlan(date=arrival.date()))
        warnings = ["Часы работы активностей нужно проверить перед поездкой"]
        if invalid_ids:
            warnings.append("Активности с просроченными источниками исключены")
        if len(all_dates) > self._max_days:
            warnings.append("Программа ограничена первыми четырьмя днями")
        return ItineraryBuildResult(
            days=tuple(days),
            unscheduled_activity_ids=tuple(
                [item.activity_id for item in (*untimed, *remaining)] + list(invalid_ids)
            ),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _dates(start: date, end: date):
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)

    @staticmethod
    def _eligible_activities(
        content: DestinationContent,
        verified_at: datetime,
    ) -> tuple[list[Activity], tuple[str, ...]]:
        evidence = {item.evidence_id: item for item in content.evidence}
        eligible: list[Activity] = []
        invalid: list[str] = []
        for activity in content.activities:
            sources = (evidence[evidence_id] for evidence_id in activity.evidence_ids)
            if all(source.is_fresh_at(verified_at) for source in sources):
                eligible.append(activity)
            else:
                invalid.append(activity.activity_id)
        return eligible, tuple(invalid)
