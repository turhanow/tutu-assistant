"""Deterministic activity scheduling inside the verified time in a destination."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from app.domain.content_models import Activity, DestinationContent
from app.domain.discovery_models import DayPlan, ScheduledActivity
from app.domain.models import RankedTripOption
from app.services.map_links import build_yandex_maps_search_url


@dataclass(frozen=True, slots=True)
class ItineraryBuildResult:
    days: tuple[DayPlan, ...]
    unscheduled_activity_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    contains_unverified_activities: bool = False

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

        target_count = self._target_activity_count(
            start_date=arrival.date(),
            end_date=departure.date(),
            max_days=self._max_days,
        )
        synthetic_count = max(0, target_count - len(activities))
        synthetic = self._generate_activities(
            content=content,
            preferred=preferred,
            count=synthetic_count,
        )
        has_unverified_activities = bool(synthetic)
        activities.extend(synthetic)
        selected = activities[:target_count]

        if (
            not combination.outbound.timezone_known
            or not combination.return_offer.timezone_known
            or arrival.tzinfo is None
            or departure.tzinfo is None
        ):
            return self._build_timezone_unknown(
                content=content,
                arrival=arrival,
                departure=departure,
                selected=selected,
                all_activities=activities,
                invalid_ids=invalid_ids,
                has_unverified_activities=has_unverified_activities,
            )

        return self._build_timezone_known(
            arrival=arrival,
            departure=departure,
            selected=selected,
            all_activities=activities,
            invalid_ids=invalid_ids,
            has_unverified_activities=has_unverified_activities,
        )

    def _build_timezone_unknown(
        self,
        content: DestinationContent,
        arrival: datetime,
        departure: datetime,
        selected: list[Activity],
        all_activities: list[Activity],
        invalid_ids: tuple[str, ...],
        has_unverified_activities: bool,
    ) -> ItineraryBuildResult:
        dates = tuple(self._dates(arrival.date(), departure.date()))[: self._max_days]
        if not dates:
            dates = (arrival.date(),)
        allocations: list[list[Activity]] = [[] for _ in dates]

        for index, activity in enumerate(selected):
            day_index = index % len(dates)
            if len(allocations[day_index]) < 3:
                allocations[day_index].append(activity)

        allocated_ids = {
            activity.activity_id for allocation in allocations for activity in allocation
        }
        remaining = [item for item in all_activities if item.activity_id not in allocated_ids]
        for day_activities in allocations:
            while len(day_activities) < 2 and remaining:
                day_activities.append(remaining.pop(0))
        if remaining:
            selected_count = {
                day_index: len(day_activities)
                for day_index, day_activities in enumerate(allocations)
            }
            for index in range(len(remaining)):
                day_index = index % len(allocations)
                if selected_count[day_index] >= 3:
                    continue
                day_activities = allocations[day_index]
                day_activities.append(remaining.pop(0))
                selected_count[day_index] += 1
                if not remaining:
                    break

        warnings = ["Часы работы активностей нужно проверить перед поездкой"]
        if has_unverified_activities:
            warnings.append("Часть программы предложена как непроверенные идеи.")
        if remaining:
            warnings.append("Программа дополнена дополнительными идеями по вашему запросу.")

        days = tuple(
            DayPlan(
                date=plan_date,
                suggestions=tuple(activities_for_day),
            )
            for plan_date, activities_for_day in zip(dates, allocations, strict=True)
        )
        allocated_ids = {
            activity.activity_id for allocation in allocations for activity in allocation
        }
        unscheduled = tuple(
            item.activity_id for item in all_activities if item.activity_id not in allocated_ids
        )
        if invalid_ids:
            warnings.append("Активности с просроченными источниками исключены")

        return ItineraryBuildResult(
            days=days,
            unscheduled_activity_ids=tuple((*unscheduled, *invalid_ids)),
            warnings=tuple(warnings),
            contains_unverified_activities=has_unverified_activities,
        )

    def _build_timezone_known(
        self,
        arrival: datetime,
        departure: datetime,
        selected: list[Activity],
        all_activities: list[Activity],
        invalid_ids: tuple[str, ...],
        has_unverified_activities: bool,
    ) -> ItineraryBuildResult:
        available_from = arrival + self._arrival_buffer
        available_until = departure - self._return_buffer
        all_dates = tuple(self._dates(arrival.date(), departure.date()))
        dates = all_dates[: self._max_days]
        if not dates:
            dates = (arrival.date(),)

        allocations: list[list[Activity]] = [[] for _ in dates]
        for index, activity in enumerate(selected):
            day_index = index % len(dates)
            if len(allocations[day_index]) < 3:
                allocations[day_index].append(activity)

        allocated_ids = {
            activity.activity_id for allocation in allocations for activity in allocation
        }
        remaining = [item for item in all_activities if item.activity_id not in allocated_ids]
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
            while len(scheduled) + len(suggestions) < 2 and remaining:
                candidate = remaining.pop(0)
                suggestions.append(candidate)
                untimed.append(candidate)
            while len(scheduled) + len(suggestions) < 3 and remaining:
                candidate = remaining.pop(0)
                suggestions.append(candidate)
                untimed.append(candidate)
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
        if has_unverified_activities:
            warnings.append("Часть программы предложена как непроверенные идеи.")
        if len(all_dates) > self._max_days:
            warnings.append("Программа ограничена первыми четырьмя днями")

        return ItineraryBuildResult(
            days=tuple(days),
            unscheduled_activity_ids=tuple(
                [item.activity_id for item in (*untimed, *remaining)] + list(invalid_ids)
            ),
            warnings=tuple(warnings),
            contains_unverified_activities=has_unverified_activities,
        )

    @staticmethod
    def _dates(start: date, end: date):
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)

    @staticmethod
    def _target_activity_count(start_date: date, end_date: date, max_days: int) -> int:
        day_count = (end_date - start_date).days + 1
        day_count = max(1, min(day_count, max_days))
        if day_count == 1:
            return 3
        return max(4, min(6, day_count * 2))

    @staticmethod
    def _generate_activities(
        content: DestinationContent,
        preferred: frozenset[str],
        count: int,
    ) -> list[Activity]:
        if count <= 0:
            return []
        templates = {
            "history": "исторический квартал и ключевые достопримечательности",
            "architecture": "архитектурная прогулка по старому центру",
            "museum": "встреча с местными музеями и галереями",
            "nature": "прогулка по живописной части города",
            "culture": "культурная зона старого города",
            "gastronomy": "местные кафе и гастрономические остановки",
            "walking": "спокойный пеший маршрут",
            "science": "научно-технические точки интереса",
            "space": "космическая тема и объекты",
            "craft": "местные мастерские и ремесленные маршруты",
            "river": "набережная и прогулка у воды",
            "relaxed": "спокойный формат на свежем воздухе",
        }
        fallback = templates["relaxed"]
        highlights = list(content.destination.activity_highlights)
        if not highlights:
            highlights = [content.destination.name]
        generated: list[Activity] = []
        for index in range(count):
            template_key = (
                next(iter(preferred), None)
                if index % 2 == 0
                else ("history" if "history" in templates else "relaxed")
            )
            pattern = templates.get(template_key, fallback)
            city = content.destination.name
            suffix = highlights[index % len(highlights)]
            activity_name = f"{pattern}: {suffix}"
            generated.append(
                Activity(
                    activity_id=f"synthetic-{content.destination.destination_id}-{index}",
                    destination_id=content.destination.destination_id,
                    name=activity_name,
                    description="Непроверенная идея: время и стоимость уточняются перед поездкой.",
                    categories=frozenset((template_key or "relaxed",)),
                    duration=timedelta(hours=1, minutes=30),
                    address=f"{city}",
                    map_url=build_yandex_maps_search_url(
                        name=activity_name,
                        city=city,
                        region=content.destination.region,
                    ),
                )
            )
        return generated

    @staticmethod
    def _eligible_activities(
        content: DestinationContent,
        verified_at: datetime,
    ) -> tuple[list[Activity], tuple[str, ...]]:
        evidence = {item.evidence_id: item for item in content.evidence}
        eligible: list[Activity] = []
        invalid: list[str] = []
        seen_names: set[str] = set()
        for activity in content.activities:
            normalized_name = " ".join(activity.name.casefold().replace("ё", "е").split())
            if normalized_name in seen_names:
                invalid.append(activity.activity_id)
                continue
            sources = (evidence[evidence_id] for evidence_id in activity.evidence_ids)
            if all(source.is_fresh_at(verified_at) for source in sources):
                eligible.append(activity)
                seen_names.add(normalized_name)
            else:
                invalid.append(activity.activity_id)
        return eligible, tuple(invalid)
