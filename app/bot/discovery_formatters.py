"""HTML-safe presentation for destination discovery and verified proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from html import escape

from app.bot.formatters import TRANSPORT_ICONS, TRANSPORT_LABELS, format_money
from app.domain.content_models import Activity
from app.domain.discovery_models import (
    CandidateShortlist,
    ProposalCopy,
    Recommendation,
    WeekendProposal,
)
from app.domain.models import RankedTripOption, SortPreference, TripCheckoutItem, TripComponent
from app.services.map_links import build_yandex_maps_search_url

TELEGRAM_SAFE_MESSAGE_LENGTH = 4_000

UNKNOWN_COST_LABELS = {
    "outbound": "дорога туда",
    "return": "дорога обратно",
    "hotel": "отель",
    "food": "питание",
    "local_transport": "транспорт по городу",
    "activities": "входные билеты и активности",
    "local_expenses_other_currency": "локальные расходы в другой валюте",
}


def format_shortlist(shortlist: CandidateShortlist) -> str:
    if not shortlist.candidates:
        return (
            "<b>Под эти условия направлений пока нет</b>\n"
            "Попробуйте изменить даты, бюджет или допустимое время в дороге."
        )
    return (
        "<b>Нашёл несколько направлений под ваше настроение</b>\n"
        "Теперь проверяю конкретные билеты, проживание и реальное время в городе."
    )


def format_proposal_sections(
    recommendations: Sequence[Recommendation],
    copies: Sequence[ProposalCopy],
    checkout_items: Mapping[str, Sequence[TripCheckoutItem]] | None = None,
) -> tuple[str, ...]:
    copy_by_id = {item.proposal_id: item for item in copies}
    sections = [
        "<b>Вот поездки, которые реально складываются</b>\n"
        "Цена и доступность актуальны на момент поиска."
    ]
    for index, recommendation in enumerate(recommendations, start=1):
        copy = copy_by_id[f"proposal_{index}"]
        destination_id = recommendation.proposal.candidate.destination.destination_id
        sections.append(
            _format_proposal(
                index,
                recommendation,
                copy,
                checkout_items=tuple((checkout_items or {}).get(destination_id, ())),
            )
        )
    return tuple(sections)


def format_program(recommendation: Recommendation) -> str:
    proposal = recommendation.proposal
    lines = [f"<b>Программа: {escape(proposal.candidate.destination.name)}</b>"]
    for day_index, day in enumerate(proposal.days, start=1):
        lines.append(f"\n<b>День {day_index} · {day.date:%d.%m}</b>")
        if not day.activities and not day.suggestions:
            lines.append("Программа на этот день пока не собрана.")
            continue
        for item in day.activities:
            qualifier = "Ориентировочно · " if not proposal.content_grounded else ""
            lines.append(
                f"{qualifier}{item.starts_at:%H:%M}–{item.ends_at:%H:%M} · "
                f"{_format_activity_link(item.activity)}"
            )
        for item in day.suggestions:
            qualifier = "Непроверенное · " if not proposal.content_grounded else ""
            lines.append(f"{qualifier}Без точного времени · {_format_activity_link(item)}")
    combination = proposal.trip_option.combination
    lines.extend(
        (
            "\n<b>Логистика</b>",
            _format_leg("Туда", combination.outbound),
            _format_leg("Обратно", combination.return_offer),
            "Выбор дороги: лучший баланс цены, времени в пути, пересадок и времени в городе.",
            f"В городе: {_format_duration(combination.metrics.time_in_city)}",
            f"Проверено: {proposal.verified_at:%d.%m.%Y %H:%M}",
            f"\n⚠️ {escape(proposal.trade_off)}",
        )
    )
    if proposal.trip_alternatives:
        lines.append("\n<b>Что ещё можно выбрать</b>")
        for option in proposal.trip_alternatives:
            lines.extend(_format_expanded_alternative(option))
    return "\n".join(lines)


def pack_html_sections(
    sections: Sequence[str],
    *,
    limit: int = TELEGRAM_SAFE_MESSAGE_LENGTH,
) -> tuple[str, ...]:
    """Pack complete HTML sections without breaking tags between Telegram messages."""
    if limit < 100:
        raise ValueError("message limit is too small")
    messages: list[str] = []
    current = ""
    for section in sections:
        if len(section) > limit:
            raise ValueError("one formatted section exceeds Telegram safe message length")
        candidate = section if not current else f"{current}\n\n{section}"
        if len(candidate) <= limit:
            current = candidate
        else:
            messages.append(current)
            current = section
    if current:
        messages.append(current)
    return tuple(messages)


def _format_proposal(
    index: int,
    recommendation: Recommendation,
    copy: ProposalCopy,
    *,
    checkout_items: Sequence[TripCheckoutItem] = (),
) -> str:
    proposal = recommendation.proposal
    combination = proposal.trip_option.combination
    label = _recommendation_label(recommendation.labels)
    anchor_activities = _proposal_anchor_activities(proposal)
    lines = [
        f"<b>{index}. {escape(copy.title)}</b>",
        f"{escape(label)}. {escape(copy.reason)}",
    ]
    if anchor_activities:
        activity_line = "Чем заняться: " + "; ".join(
            _format_activity_link(item) for item in anchor_activities
        )
        if not proposal.content_grounded:
            activity_line += " · идеи AI, детали пока не проверены"
        lines.append(activity_line)
    checkout_by_component = {item.component: item for item in checkout_items}
    lines.extend(
        (
            _format_leg(
                "Туда",
                combination.outbound,
                checkout_by_component.get(TripComponent.OUTBOUND),
            ),
            _format_leg(
                "Обратно",
                combination.return_offer,
                checkout_by_component.get(TripComponent.RETURN),
            ),
            "Выбор дороги: лучший баланс цены, времени в пути, пересадок и времени в городе.",
            f"В дороге: {_format_duration(combination.metrics.total_travel_duration)}",
            f"В городе: {_format_duration(combination.metrics.time_in_city)}",
        )
    )
    if combination.hotel is not None:
        hotel_checkout = checkout_by_component.get(TripComponent.HOTEL)
        map_url = build_yandex_maps_search_url(
            name=combination.hotel.name,
            city=proposal.candidate.destination.name,
            region=proposal.candidate.destination.region,
        )
        hotel_name = _html_link(escape(combination.hotel.name), map_url)
        booking_url = str(hotel_checkout.link.url) if hotel_checkout is not None else None
        if combination.hotel.room_name:
            room_name = escape(combination.hotel.room_name)
            if booking_url is not None:
                room_name = _html_link(room_name, booking_url)
            booking = f" · номер: {room_name}"
        elif booking_url is not None:
            booking = f" · {_html_link('забронировать', booking_url)}"
        else:
            booking = ""
        lines.append(f"🏨 Отель: {hotel_name}{booking}")
    else:
        lines.append("🏨 Отель: не нужен")
    cost = proposal.cost
    if cost.confirmed_total is not None and cost.confirmed_currency is not None:
        lines.append(
            "💳 Предварительная стоимость: "
            + format_money(cost.confirmed_total, cost.confirmed_currency)
        )
    else:
        lines.append("💳 Предварительная стоимость: пока неизвестна")
    if cost.estimated is not None:
        lines.append(
            "Дополнительно, оценка: "
            + format_money(cost.estimated.minimum, cost.estimated.currency)
            + "–"
            + format_money(cost.estimated.maximum, cost.estimated.currency)
        )
    if cost.unknown_components:
        unknown = ", ".join(
            escape(UNKNOWN_COST_LABELS.get(item, item)) for item in cost.unknown_components
        )
        lines.append(f"Пока не учтено: {unknown}")
    comparison = _format_trip_comparison(proposal)
    if comparison:
        lines.append(comparison)
    lines.append(f"⚠️ {escape(copy.trade_off)}")
    return "\n".join(lines)


def _format_activity_link(activity: Activity) -> str:
    name = escape(activity.name)
    if activity.map_url is None:
        return name
    return f'<a href="{escape(str(activity.map_url), quote=True)}">{name}</a>'


def _proposal_anchor_activities(proposal: WeekendProposal) -> tuple[Activity, ...]:
    candidates = [item.activity for day in proposal.days for item in day.activities]
    candidates.extend(item for day in proposal.days for item in day.suggestions)
    candidates.extend(proposal.suggested_activities)
    selected: list[Activity] = []
    seen: set[str] = set()
    for activity in candidates:
        if activity.activity_id in seen:
            continue
        selected.append(activity)
        seen.add(activity.activity_id)
        if len(selected) == 2:
            break
    return tuple(selected)


def _recommendation_label(labels: frozenset[str]) -> str:
    ordered = [
        text
        for key, text in (
            ("best_match", "лучше всего подходит под запрос"),
            ("lowest_confirmed_cost", "самый бюджетный из проверенных"),
            ("most_time_in_city", "больше времени останется на город"),
        )
        if key in labels
    ]
    if len(ordered) > 1:
        return "Сильный вариант сразу по нескольким причинам: " + ", ".join(ordered)
    return ordered[0].capitalize() if ordered else "Проверенный вариант"


def _format_leg(label, offer, checkout_item: TripCheckoutItem | None = None) -> str:
    transfers = (
        "пересадки не указаны"
        if offer.transfers is None
        else "без пересадок"
        if offer.transfers == 0
        else f"пересадок: {offer.transfers}"
    )
    service = TRANSPORT_LABELS[offer.mode]
    if offer.service_number:
        service += f" № {escape(offer.service_number)}"
    if offer.carrier:
        service += f" · {escape(offer.carrier)}"
    link_suffix = ""
    if checkout_item is not None:
        url = str(checkout_item.link.url)
        if checkout_item.link.kind in {
            "direct_offer",
            "hotel_page",
            "deeplink",
            "checkout_deeplink",
        }:
            service = _html_link(service, url)
        elif checkout_item.link.kind == "schedule_url":
            link_suffix = f" · {_html_link('расписание на Tutu', url)}"
        else:
            link_suffix = f" · {_html_link('варианты на Tutu', url)}"
    price = (
        f" · {format_money(offer.price, offer.currency or '')}" if offer.price is not None else ""
    )
    return (
        f"{TRANSPORT_ICONS[offer.mode]} {label}: {service} · "
        f"{_format_datetime(offer.departure_at)} → {_format_datetime(offer.arrival_at)} · "
        f"{transfers}{price}{link_suffix}"
    )


def _html_link(label: str, url: str) -> str:
    return f'<a href="{escape(url, quote=True)}">{label}</a>'


def _format_datetime(value: datetime) -> str:
    return f"{value:%d.%m, %H:%M}"


def _format_duration(value: timedelta) -> str:
    total_minutes = max(int(value.total_seconds() // 60), 0)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours} ч {minutes} мин" if minutes else f"{hours} ч"


def _format_trip_comparison(proposal: WeekendProposal) -> str | None:
    alternatives = proposal.trip_alternatives
    parts: list[str] = []
    for preference, label in (
        (SortPreference.CHEAPEST, "дешевле"),
        (SortPreference.FASTEST, "быстрее"),
    ):
        option = next((item for item in alternatives if preference in item.labels), None)
        if option is None:
            continue
        total = option.combination.price.known_total
        currency = option.combination.price.currency
        if total is not None:
            parts.append(f"{label} за {format_money(total, currency)}")
        else:
            parts.append(f"{label} — цена уточняется")
    if not parts:
        return None
    return "Сравнить варианты: " + "; ".join(parts)


def _format_expanded_alternative(option: RankedTripOption) -> tuple[str, ...]:
    labels = []
    if SortPreference.CHEAPEST in option.labels:
        labels.append("бюджетный")
    if SortPreference.BALANCED in option.labels:
        labels.append("оптимальный")
    if SortPreference.FASTEST in option.labels:
        labels.append("быстрый")
    title = ", ".join(labels) if labels else "ещё один подходящий"
    combination = option.combination
    lines = [f"<b>{escape(title.capitalize())}</b>"]
    lines.append(_format_leg("Туда", combination.outbound))
    lines.append(_format_leg("Обратно", combination.return_offer))
    if combination.hotel is not None:
        lines.append(f"🏨 Отель: {escape(combination.hotel.name)}")
    total = combination.price.known_total
    if total is not None:
        lines.append(
            "💳 Предварительная стоимость: "
            + format_money(total, combination.price.currency)
        )
    return tuple(lines)
