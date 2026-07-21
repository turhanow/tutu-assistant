"""Deterministic completion of an extracted discovery draft."""

from __future__ import annotations

from datetime import date

from app.domain.discovery_models import (
    DateFlexibility,
    DateRange,
    DiscoveryDraft,
    DiscoveryRequest,
)
from app.domain.models import HotelMode, TransportPreferences, TravelerComposition

CRITICAL_FIELDS = ("origin", "dates", "motives")
OPTIONAL_CLARIFICATION_FIELDS = ("budget", "road_tolerance", "travelers")


class DiscoveryInputError(ValueError):
    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


def missing_discovery_fields(draft: DiscoveryDraft) -> tuple[str, ...]:
    missing: list[str] = []
    if draft.origin is None:
        missing.append("origin")
    if draft.departure_date is None or draft.return_date is None:
        missing.append("dates")
    if not draft.experience.motives and not draft.experience.interests:
        missing.append("motives")
    if draft.hotel_mode is None:
        missing.append("hotel_mode")
    return tuple(missing)


def useful_optional_fields(draft: DiscoveryDraft) -> tuple[str, ...]:
    fields: list[str] = []
    if draft.budget is None:
        fields.append("budget")
    tolerance = draft.experience.road_tolerance
    if (
        tolerance.max_one_way_duration is None
        and tolerance.max_transfers is None
        and tolerance.allow_night_travel is None
    ):
        fields.append("road_tolerance")
    if draft.adults is None:
        fields.append("travelers")
    return tuple(fields)


def build_discovery_request(
    draft: DiscoveryDraft,
    *,
    today: date,
) -> DiscoveryRequest:
    missing = missing_discovery_fields(draft)
    if missing:
        raise DiscoveryInputError(
            f"Не заполнены обязательные поля: {', '.join(missing)}",
            missing[0],
        )
    assert draft.origin is not None
    assert draft.departure_date is not None
    assert draft.return_date is not None
    if draft.departure_date < today:
        raise DiscoveryInputError("Дата начала поездки уже прошла.", "dates")
    if draft.return_date < draft.departure_date:
        raise DiscoveryInputError(
            "Дата возвращения не может быть раньше даты отправления.",
            "dates",
        )
    if (draft.adults or 1) > 2 or draft.children > 0 or draft.rooms > 1:
        raise DiscoveryInputError(
            "Сейчас поддерживаются только 1–2 взрослых, без детей и один номер. "
            "Состав поездки не будет изменён автоматически.",
            "travelers",
        )
    if draft.departure_date == draft.return_date and draft.hotel_mode is HotelMode.REQUIRED:
        raise DiscoveryInputError(
            "Для поездки в тот же день нельзя добавить ночёвку. "
            "Выберите вариант без отеля или измените даты.",
            "hotel_mode",
        )
    try:
        dates = DateRange(
            start=draft.departure_date,
            end=draft.return_date,
            flexibility=draft.date_flexibility or DateFlexibility.FIXED,
        )
        return DiscoveryRequest(
            origin=draft.origin,
            dates=dates,
            travelers=TravelerComposition(
                adults=draft.adults or 1,
                children=draft.children,
                rooms=draft.rooms,
            ),
            budget=draft.budget,
            currency=draft.currency,
            hotel_mode=draft.hotel_mode,
            experience=draft.experience,
            transport=TransportPreferences(
                allowed_modes=draft.allowed_modes,
                max_transfers=draft.experience.road_tolerance.max_transfers,
            ),
        )
    except ValueError as error:
        raise DiscoveryInputError(str(error), "dates") from error
