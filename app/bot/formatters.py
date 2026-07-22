"""Pure, HTML-safe Telegram message formatting."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from html import escape

from app.domain.models import (
    HotelMode,
    OfferDetails,
    RankedTripOption,
    SortPreference,
    TransportMode,
    TripComponent,
    TripRequest,
    TripSearchResult,
)

MONTHS = (
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)

HOTEL_LABELS = {
    HotelMode.REQUIRED: "нужен",
    HotelMode.FORBIDDEN: "не нужен",
    HotelMode.OPTIONAL: "необязателен",
}

COMPONENT_LABELS = {
    TripComponent.OUTBOUND: "Поездка туда",
    TripComponent.RETURN: "Поездка обратно",
    TripComponent.HOTEL: "Отель",
}

TRANSPORT_LABELS = {
    TransportMode.AVIA: "самолёт",
    TransportMode.RAIL: "поезд",
    TransportMode.BUS: "автобус",
    TransportMode.ETRAIN: "электричка",
    TransportMode.MIXED: "комбинированный маршрут",
    TransportMode.UNKNOWN: "другой транспорт",
}

TRANSPORT_ICONS = {
    TransportMode.AVIA: "✈️",
    TransportMode.RAIL: "🚆",
    TransportMode.BUS: "🚌",
    TransportMode.ETRAIN: "🚆",
    TransportMode.MIXED: "🧭",
    TransportMode.UNKNOWN: "🧭",
}


def format_confirmation(request: TripRequest) -> str:
    modes = (
        ", ".join(TRANSPORT_LABELS[mode] for mode in sorted(request.transport.allowed_modes))
        or "любые"
    )
    budget = (
        format_money(request.budget, request.currency)
        if request.budget is not None
        else "не указан"
    )
    lines = [
        "<b>Всё верно?</b>\n"
        f"Маршрут: {escape(request.origin)} → {escape(request.destination)}\n"
        f"Даты: {_format_date(request.departure_date, year=True)} — "
        f"{_format_date(request.return_date, year=True)}\n"
        f"Пассажиры: {_plural(request.travelers.adults, 'взрослый', 'взрослых', 'взрослых')}\n"
        f"Отель: {HOTEL_LABELS[request.hotel.mode]}\n"
        f"Транспорт: {escape(modes)}\n"
        f"Бюджет: {budget}"
    ]
    if request.transport.max_transfers is not None:
        lines.append(f"Пересадки: не более {request.transport.max_transfers}")
    if request.transport.departure_window is not None:
        lines.append("Отправление туда: " + _format_window(request.transport.departure_window))
    if request.transport.return_window is not None:
        lines.append("Отправление обратно: " + _format_window(request.transport.return_window))
    hotel_requirements = _hotel_requirements(request)
    if hotel_requirements:
        lines.append("Требования к отелю: " + ", ".join(hotel_requirements))
    if request.event is not None:
        event = f"{_format_date(request.event.date, year=True)}"
        if request.event.start_time is not None:
            event += f", с {request.event.start_time:%H:%M}"
        if request.event.end_time is not None:
            end_date = request.event.end_date or request.event.date
            if end_date != request.event.date:
                event += f" до {_format_date(end_date)} {request.event.end_time:%H:%M}"
            else:
                event += f" до {request.event.end_time:%H:%M}"
        lines.append("Событие: " + event)
    sort_labels = {
        SortPreference.CHEAPEST: "минимальная цена",
        SortPreference.FASTEST: "минимум времени в дороге",
        SortPreference.BALANCED: "баланс цены и удобства",
    }
    lines.append("Приоритет: " + sort_labels[request.sort])
    return "\n".join(lines)


def format_results(result: TripSearchResult) -> str:
    if not result.options:
        reason = result.failures[0].user_message if result.failures else "Варианты не найдены"
        return f"<b>На этих условиях поездка пока не складывается</b>\n{escape(reason)}"
    cards = ["<b>Вот что складывается на эти даты</b>"]
    for index, option in enumerate(result.options, start=1):
        cards.append(_format_option(index, option))
    if result.failures:
        cards.append("Часть данных недоступна: " + escape(result.failures[0].user_message))
    return "\n\n".join(cards)


def format_details(component: TripComponent, details: OfferDetails) -> str:
    title = _localized_details_title(component, details.title)
    lines = [
        f"<b>{COMPONENT_LABELS[component]}: {escape(title)}</b>",
    ]
    if details.check_in_time is not None:
        lines.append(f"Заезд после {details.check_in_time:%H:%M}")
    if details.check_out_time is not None:
        lines.append(f"Выезд до {details.check_out_time:%H:%M}")
    if details.amenities:
        amenities = ", ".join(escape(item) for item in details.amenities[:6])
        lines.append(f"Удобства: {amenities}")
    lines.extend(escape(item) for item in details.facts[:6])
    if details.rooms:
        room = details.rooms[0]
        room_text = escape(room.name)
        if room.rates and room.rates[0].price is not None:
            rate = room.rates[0]
            room_text += " — " + format_money(rate.price, rate.currency or "")
        lines.append(f"Пример доступного тарифа: {room_text}")
    lines.append("Наличие и итоговая цена подтверждаются на Tutu.")
    return "\n".join(lines)


def _format_option(index: int, option: RankedTripOption) -> str:
    combination = option.combination
    price = combination.price
    if price.known_total is None:
        price_text = "стоимость уточняется"
    else:
        qualifier = "от " if price.missing_components else ""
        price_text = qualifier + format_money(price.known_total, price.currency or "")
    title, explanation = format_ranking(option.labels)
    lines = [
        f"<b>{index}. {title}</b>",
        explanation,
        _format_leg("Туда", combination.outbound),
        _format_leg("Обратно", combination.return_offer),
        f"🕒 В дороге: {_format_duration(combination.metrics.total_travel_duration)}",
        f"В городе: {_format_duration(combination.metrics.time_in_city)}",
    ]
    if combination.hotel is not None and combination.stay is not None:
        nights = _plural(combination.stay.nights, "ночь", "ночи", "ночей")
        lines.extend(
            (
                f"🏨 Отель: {escape(combination.hotel.name)}",
                f"Проживание: {_format_date(combination.stay.check_in)} — "
                f"{_format_date(combination.stay.check_out)}, {nights}",
            )
        )
    else:
        lines.append("🏨 Отель: не нужен")
    lines.append(f"💳 Известная стоимость: {price_text}")
    included = _known_price_components(combination)
    if included:
        lines.append("В известную сумму входят: " + ", ".join(included))
    if price.missing_components:
        missing = ", ".join(_component_name(item) for item in price.missing_components)
        lines.append(f"Пока не учтено из бронируемых компонентов: {escape(missing)}")
    lines.append("Не включено: питание, городской транспорт и активности")
    lines.extend(f"⚠️ {escape(warning)}" for warning in combination.warnings)
    return "\n".join(lines)


def format_ranking(labels: frozenset[SortPreference]) -> tuple[str, str]:
    if labels == {
        SortPreference.CHEAPEST,
        SortPreference.FASTEST,
        SortPreference.BALANCED,
    }:
        return (
            "Оптимальный и самый бюджетный",
            "Это самый бюджетный и быстрый вариант, при этом лучший по балансу условий.",
        )
    descriptions = {
        SortPreference.CHEAPEST: "самый бюджетный",
        SortPreference.FASTEST: "требует меньше всего времени в дороге",
        SortPreference.BALANCED: "даёт лучший баланс цены и удобства",
    }
    ordered = [
        descriptions[label]
        for label in (
            SortPreference.CHEAPEST,
            SortPreference.FASTEST,
            SortPreference.BALANCED,
        )
        if label in labels
    ]
    if len(ordered) == 2:
        pair_title = {
            frozenset({SortPreference.CHEAPEST, SortPreference.BALANCED}): (
                "Оптимальный и самый бюджетный"
            ),
            frozenset({SortPreference.CHEAPEST, SortPreference.FASTEST}): (
                "Самый бюджетный и быстрый"
            ),
            frozenset({SortPreference.FASTEST, SortPreference.BALANCED}): ("Оптимальный и быстрый"),
        }[labels]
        return (
            pair_title,
            f"Этот вариант одновременно {ordered[0]} и {ordered[1]} среди найденных.",
        )
    if ordered:
        label = ordered[0]
        title = {
            SortPreference.CHEAPEST: "Самый бюджетный",
            SortPreference.FASTEST: "Минимум времени в дороге",
            SortPreference.BALANCED: "Лучший баланс",
        }[next(iter(labels))]
        return (title, f"Этот вариант {label} среди найденных.")
    return ("Вариант поездки", "Подходит под подтверждённые параметры.")


def _localized_details_title(component: TripComponent, title: str) -> str:
    normalized = title.strip().casefold()
    generic_titles = {
        "bus offer": "Автобусный билет",
        "rail offer": "Билет на поезд",
        "train offer": "Билет на поезд",
        "railway offer": "Билет на поезд",
        "etrain offer": "Билет на электричку",
        "flight offer": "Авиабилет",
        "avia offer": "Авиабилет",
        "hotel offer": "Вариант проживания",
    }
    if normalized in generic_titles:
        return generic_titles[normalized]
    if normalized in {"offer", "transport offer", "unknown offer"}:
        return COMPONENT_LABELS[component]
    return title


def format_money(amount: Decimal, currency: str) -> str:
    quantized = amount.quantize(Decimal("0.01"))
    rendered = f"{quantized:,.0f}" if quantized == quantized.to_integral() else f"{quantized:,.2f}"
    rendered = rendered.replace(",", " ").replace(".", ",")
    suffix = "₽" if currency == "RUB" else escape(currency)
    return f"{rendered} {suffix}".rstrip()


def _format_leg(label: str, offer) -> str:
    transfers = (
        "пересадки не указаны"
        if offer.transfers is None
        else "без пересадок"
        if offer.transfers == 0
        else _plural(offer.transfers, "пересадка", "пересадки", "пересадок")
    )
    return (
        f"{TRANSPORT_ICONS[offer.mode]} {label}: {TRANSPORT_LABELS[offer.mode]} · "
        f"{_format_datetime(offer.departure_at)} → {_format_datetime(offer.arrival_at)} · "
        f"{transfers}"
    )


def _format_datetime(value: datetime) -> str:
    return f"{_format_date(value.date())}, {value:%H:%M}"


def _format_date(value: date, *, year: bool = False) -> str:
    suffix = f" {value.year}" if year else ""
    return f"{value.day} {MONTHS[value.month]}{suffix}"


def _format_duration(value: timedelta) -> str:
    total_minutes = max(int(value.total_seconds() // 60), 0)
    hours, minutes = divmod(total_minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} ч")
    if minutes or not parts:
        parts.append(f"{minutes} мин")
    return " ".join(parts)


def _plural(number: int, one: str, few: str, many: str) -> str:
    last_two = number % 100
    last = number % 10
    word = (
        one
        if last == 1 and last_two != 11
        else few
        if 2 <= last <= 4 and not 12 <= last_two <= 14
        else many
    )
    return f"{number} {word}"


def _format_window(window) -> str:
    if window.from_time is not None and window.to_time is not None:
        return f"с {window.from_time:%H:%M} до {window.to_time:%H:%M}"
    if window.from_time is not None:
        return f"после {window.from_time:%H:%M}"
    return f"до {window.to_time:%H:%M}"


def _hotel_requirements(request: TripRequest) -> list[str]:
    preferences = request.hotel
    values: list[str] = []
    if preferences.stars_min is not None:
        values.append(f"от {preferences.stars_min}★")
    if preferences.rating_min is not None:
        values.append(f"рейтинг от {preferences.rating_min}")
    if preferences.free_cancellation_required:
        values.append("бесплатная отмена")
    if preferences.meal_preferences:
        values.append("питание: " + ", ".join(sorted(preferences.meal_preferences)))
    if preferences.required_amenities:
        values.append("удобства: " + ", ".join(sorted(preferences.required_amenities)))
    return values


def _component_name(value: str) -> str:
    return {
        "outbound": "дорога туда",
        "return": "дорога обратно",
        "hotel": "отель",
    }.get(value, value)


def _known_price_components(combination) -> list[str]:
    values: list[str] = []
    if combination.outbound.price is not None:
        values.append("дорога туда")
    if combination.return_offer.price is not None:
        values.append("дорога обратно")
    if combination.hotel is not None and combination.hotel.price is not None:
        values.append("отель")
    return values
