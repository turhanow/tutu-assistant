"""Deterministic completion and validation of an LLM-produced trip draft."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from app.domain.models import (
    EventConstraint,
    HotelMode,
    HotelPreferences,
    ParsedTripDraft,
    SortPreference,
    TransportMode,
    TransportPreferences,
    TravelerComposition,
    TripRequest,
)
from app.services.known_input_guardrails import normalize_city_answer

REQUIRED_FIELDS = (
    "origin",
    "destination",
    "departure_date",
    "return_date",
    "hotel_mode",
)


class DraftInputError(ValueError):
    """A structured-form answer cannot be mapped to the requested field."""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


def next_missing_field(draft: ParsedTripDraft) -> str | None:
    for field in REQUIRED_FIELDS:
        if getattr(draft, field) is not None:
            continue
        if field == "hotel_mode" and _is_overnight_trip(draft.departure_date, draft.return_date):
            continue
        return field
    return None


def apply_form_answer(
    draft: ParsedTripDraft,
    field: str,
    raw_value: str,
    *,
    today: date | None = None,
) -> ParsedTripDraft:
    value = raw_value.strip()
    if field in {"origin", "destination"}:
        try:
            city = normalize_city_answer(value)
        except ValueError as error:
            raise DraftInputError(str(error)) from error
        return draft.model_copy(update={field: city})
    if field in {"departure_date", "return_date"}:
        return draft.model_copy(update={field: _parse_date(value, today=today)})
    if field == "event_date":
        return draft.model_copy(update={field: _parse_date(value, today=today)})
    if field == "hotel_mode":
        normalized = value.casefold().replace("ё", "е").strip(" .,!?:;-")
        aliases = {
            "да": HotelMode.REQUIRED,
            "нужен": HotelMode.REQUIRED,
            "required": HotelMode.REQUIRED,
            "нет": HotelMode.FORBIDDEN,
            "не нужен": HotelMode.FORBIDDEN,
            "forbidden": HotelMode.FORBIDDEN,
            "необязательно": HotelMode.OPTIONAL,
            "опционально": HotelMode.OPTIONAL,
            "optional": HotelMode.OPTIONAL,
            "да, нужен отель": HotelMode.REQUIRED,
            "да нужен отель": HotelMode.REQUIRED,
            "нужен отель": HotelMode.REQUIRED,
            "нет, отель не нужен": HotelMode.FORBIDDEN,
            "нет отель не нужен": HotelMode.FORBIDDEN,
            "отель не нужен": HotelMode.FORBIDDEN,
            "можно без отеля": HotelMode.OPTIONAL,
            "отель необязателен": HotelMode.OPTIONAL,
        }
        if normalized not in aliases:
            raise DraftInputError("Ответьте: да, нет или необязательно.")
        return draft.model_copy(update={field: aliases[normalized]})
    if field == "adults":
        if value not in {"1", "2"}:
            raise DraftInputError("Пока поддерживается один или два взрослых.")
        return draft.model_copy(update={field: int(value)})
    if field == "budget":
        if value.casefold() in {"нет", "не указан", "без бюджета"}:
            return draft.model_copy(update={field: None})
        try:
            budget = Decimal(value.replace(" ", "").replace(",", "."))
        except InvalidOperation as error:
            raise DraftInputError("Введите сумму в рублях, например 25 000, или «нет».") from error
        if budget < 0:
            raise DraftInputError("Бюджет не может быть отрицательным.")
        return draft.model_copy(update={field: budget})
    if field == "allowed_modes":
        normalized = value.casefold().replace("ё", "е")
        if normalized in {"любые", "любой", "не важно"}:
            return draft.model_copy(update={field: frozenset()})
        aliases = {
            "поезд": TransportMode.RAIL,
            "самолет": TransportMode.AVIA,
            "автобус": TransportMode.BUS,
            "электричка": TransportMode.ETRAIN,
        }
        selected = {mode for label, mode in aliases.items() if label in normalized}
        if not selected:
            raise DraftInputError("Укажите: поезд, самолёт, автобус, электричка или любые.")
        return draft.model_copy(update={field: frozenset(selected)})
    if field == "sort":
        normalized = value.casefold().replace("ё", "е")
        aliases = {
            "дешевле": SortPreference.CHEAPEST,
            "цена": SortPreference.CHEAPEST,
            "быстрее": SortPreference.FASTEST,
            "времен": SortPreference.FASTEST,
            "баланс": SortPreference.BALANCED,
            "удобство": SortPreference.BALANCED,
        }
        selected = next((item for label, item in aliases.items() if label in normalized), None)
        if selected is None:
            raise DraftInputError("Укажите: дешевле, меньше времени в дороге или баланс.")
        return draft.model_copy(update={field: selected})
    raise DraftInputError("Неизвестное поле формы.")


def build_trip_request(
    draft: ParsedTripDraft,
    *,
    today: date | None = None,
) -> TripRequest:
    missing = next_missing_field(draft)
    if missing is not None:
        raise DraftInputError(f"Не заполнено обязательное поле: {missing}")
    if (
        draft.origin
        and draft.destination
        and draft.origin.casefold() == draft.destination.casefold()
    ):
        raise DraftInputError(
            "Город назначения должен отличаться от города отправления.", "destination"
        )
    if draft.departure_date and draft.return_date and draft.return_date < draft.departure_date:
        raise DraftInputError(
            "Дата возвращения не может быть раньше даты отправления.", "return_date"
        )
    if today is not None and draft.departure_date and draft.departure_date < today:
        raise DraftInputError(
            "Дата отправления уже прошла. Укажите сегодняшнюю или будущую дату.",
            "departure_date",
        )
    if (
        draft.departure_date
        and draft.return_date
        and (draft.return_date - draft.departure_date).days > 3
    ):
        raise DraftInputError(
            "Поездка может длиться не более четырёх календарных дней.", "return_date"
        )
    if draft.departure_date == draft.return_date and draft.hotel_mode is HotelMode.REQUIRED:
        raise DraftInputError(
            "Для поездки в тот же день нет ночи для проживания. "
            "Выберите «отель не нужен» или перенесите дату возвращения.",
            "hotel_mode",
        )
    effective_return_date = (
        draft.departure_date if draft.hotel_mode is HotelMode.FORBIDDEN else draft.return_date
    )
    if (draft.adults or 1) > 2 or draft.children > 0 or draft.rooms > 1:
        raise DraftInputError(
            "Сейчас поддерживаются только 1–2 взрослых, без детей и один номер. "
            "Я не буду автоматически менять состав поездки — укажите 1 или 2 взрослых "
            "либо продолжите поиск напрямую на Tutu.",
            "adults",
        )
    if (
        draft.event_date
        and draft.departure_date
        and draft.return_date
        and not (draft.departure_date <= draft.event_date <= draft.return_date)
    ):
        raise DraftInputError("Дата события должна входить в даты поездки.", "event_date")
    event = None
    if draft.event_date is not None:
        event = EventConstraint(
            date=draft.event_date,
            start_time=draft.event_start_time,
            end_time=draft.event_end_time,
            end_date=draft.event_end_date,
        )
    return TripRequest(
        origin=draft.origin,
        destination=draft.destination,
        departure_date=draft.departure_date,
        return_date=effective_return_date,
        travelers=TravelerComposition(
            adults=draft.adults or 1,
            children=draft.children,
            rooms=draft.rooms,
        ),
        budget=draft.budget,
        currency=draft.currency,
        transport=TransportPreferences(
            allowed_modes=draft.allowed_modes,
            max_transfers=draft.max_transfers,
            departure_window=draft.departure_window,
            return_window=draft.return_window,
        ),
        hotel=HotelPreferences(
            mode=(
                HotelMode.REQUIRED
                if _is_overnight_trip(draft.departure_date, effective_return_date)
                else draft.hotel_mode or HotelMode.OPTIONAL
            ),
            stars_min=draft.hotel_stars_min,
            rating_min=draft.hotel_rating_min,
            meal_preferences=draft.hotel_meals,
            free_cancellation_required=draft.hotel_free_cancellation,
            required_amenities=draft.hotel_amenities,
        ),
        event=event,
        sort=draft.sort or SortPreference.BALANCED,
    )


def _is_overnight_trip(departure_date: date | None, return_date: date | None) -> bool:
    return departure_date is not None and return_date is not None and return_date > departure_date


def _parse_date(value: str, *, today: date | None = None) -> date:
    normalized = value.casefold().replace("ё", "е").strip()
    if today is not None and normalized in {"сегодня", "завтра", "послезавтра"}:
        offset = {"сегодня": 0, "завтра": 1, "послезавтра": 2}[normalized]
        return date.fromordinal(today.toordinal() + offset)
    for pattern in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    months = {
        "января": 1,
        "февраля": 2,
        "марта": 3,
        "апреля": 4,
        "мая": 5,
        "июня": 6,
        "июля": 7,
        "августа": 8,
        "сентября": 9,
        "октября": 10,
        "ноября": 11,
        "декабря": 12,
    }
    parts = normalized.split()
    if len(parts) in {2, 3} and parts[0].isdigit() and parts[1] in months:
        year = int(parts[2]) if len(parts) == 3 and parts[2].isdigit() else None
        if year is None and today is not None:
            year = today.year
        if year is not None:
            try:
                parsed = date(year, months[parts[1]], int(parts[0]))
            except ValueError as error:
                raise DraftInputError("Такой календарной даты нет.") from error
            if len(parts) == 2 and today is not None and parsed < today:
                parsed = parsed.replace(year=year + 1)
            return parsed
    raise DraftInputError("Введите дату как ДД.ММ.ГГГГ, например «25.07.2026» или «25 июля 2026».")
