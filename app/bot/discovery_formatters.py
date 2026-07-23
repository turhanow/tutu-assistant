"""HTML-safe presentation for destination discovery and verified proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from html import escape
from html.parser import HTMLParser

from app.bot.formatters import TRANSPORT_ICONS, TRANSPORT_LABELS, format_money
from app.domain.content_models import Activity, activity_place_identities
from app.domain.discovery_models import (
    CandidateShortlist,
    ProposalCopy,
    Recommendation,
    WeekendProposal,
)
from app.domain.models import (
    HotelOffer,
    RankedTripOption,
    SortPreference,
    TripCheckoutItem,
    TripCombination,
    TripComponent,
)
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


def format_program(
    recommendation: Recommendation,
    checkout_items: Sequence[TripCheckoutItem] = (),
) -> str:
    proposal = recommendation.proposal
    destination = proposal.candidate.destination
    combination = proposal.trip_option.combination
    checkout_by_component = _checkout_items_by_component(
        checkout_items,
        proposal.trip_option,
        allow_unscoped=True,
    )
    lines = [
        f"<b>План на 2 дня: {escape(destination.name)}</b>",
        _format_total_cost(proposal),
        _format_trip_length(combination),
    ]
    for day_index, day in enumerate(proposal.days, start=1):
        lines.append(f"\n<b>День {day_index} · {day.date:%d.%m}</b>")
        if not day.activities and not day.suggestions:
            lines.append("Программа на этот день пока не собрана.")
            continue
        for item in day.activities:
            qualifier = "Ориентировочно · " if not proposal.content_grounded else ""
            lines.append(
                f"{qualifier}{item.starts_at:%H:%M}–{item.ends_at:%H:%M} · "
                f"{_format_activity_link(item.activity, destination)}"
            )
        for item in day.suggestions:
            qualifier = "Непроверенное · " if not proposal.content_grounded else ""
            lines.append(
                f"{qualifier}Без точного времени · {_format_activity_link(item, destination)}"
            )
    if combination.hotel is not None:
        lines.extend(
            (
                "\n<b>Проживание</b>",
                _format_hotel_plan(
                    combination.hotel,
                    destination_name=destination.name,
                    destination_region=destination.region,
                    checkout_item=checkout_by_component.get(TripComponent.HOTEL),
                ),
            )
        )
    taxi = destination.taxi_available
    taxi_text = (
        "да, но доступность и цену лучше проверить в приложении"
        if taxi is True
        else "нет"
        if taxi is False
        else "лучше проверить перед поездкой"
    )
    lines.append(f"🚕 Такси в городе: {taxi_text}")
    lines.extend(
        (
            "\n<b>Логистика</b>",
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
            f"В городе: {_format_duration(combination.metrics.time_in_city)}",
            f"Проверено: {proposal.verified_at:%d.%m.%Y %H:%M}",
            f"\n⚠️ {escape(proposal.trade_off)}",
        )
    )
    if proposal.trip_alternatives:
        lines.append("\n<b>Что ещё можно выбрать</b>")
        for option in proposal.trip_alternatives:
            lines.extend(
                _format_expanded_alternative(
                    option,
                    (proposal.trip_option, *proposal.trip_alternatives),
                    checkout_items=checkout_items,
                    destination_name=destination.name,
                    destination_region=destination.region,
                )
            )
    return "\n".join(lines)


def format_details(
    recommendation: Recommendation,
    checkout_items: Sequence[TripCheckoutItem] = (),
) -> str:
    proposal = recommendation.proposal
    destination = proposal.candidate.destination
    combination = proposal.trip_option.combination
    checkout_by_component = _checkout_items_by_component(
        checkout_items,
        proposal.trip_option,
        allow_unscoped=True,
    )
    city_url = build_yandex_maps_search_url(
        name=destination.name,
        region=destination.region,
    )
    lines = [
        f"<b>Подробнее: {_html_link(escape(destination.name), city_url)}</b>",
        _format_total_cost(proposal),
        _format_trip_length(combination),
        escape(
            destination.full_description
            or destination.short_description
            or _fallback_city_description(proposal)
        ),
        "Фото и отзывы — по ссылке в названии города.",
        "\n<b>Что посмотреть</b>",
    ]
    for activity in _proposal_anchor_activities(proposal)[:3]:
        description = f" — {escape(activity.description)}" if activity.description else ""
        lines.append(f"• {_format_activity_link(activity, destination)}{description}")
    lines.extend(
        (
            "\n<b>Дорога</b>",
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
            _format_transport_total(combination),
            f"Общее время в транспорте: "
            f"{_format_duration(combination.metrics.total_travel_duration)}",
        )
    )
    if combination.hotel is not None:
        lines.extend(
            (
                "\n<b>Отель</b>",
                _format_hotel_details(
                    combination.hotel,
                    destination_name=destination.name,
                    destination_region=destination.region,
                    checkout_item=checkout_by_component.get(TripComponent.HOTEL),
                ),
            )
        )
    else:
        lines.append("\n🏨 Отель не нужен: поездка без ночёвки.")
    lines.append(f"\n⚠️ {escape(proposal.trade_off)}")
    return "\n".join(lines)


def pack_html_sections(
    sections: Sequence[str],
    *,
    limit: int = TELEGRAM_SAFE_MESSAGE_LENGTH,
) -> tuple[str, ...]:
    """Pack HTML by Telegram's post-entity text limit, preserving complete lines."""
    if limit < 100:
        raise ValueError("message limit is too small")
    messages: list[str] = []
    current = ""
    for section in sections:
        for fragment in _split_html_section(section, limit=limit):
            candidate = fragment if not current else f"{current}\n\n{fragment}"
            if _telegram_text_length(candidate) <= limit:
                current = candidate
            else:
                if current:
                    messages.append(current)
                current = fragment
    if current:
        messages.append(current)
    return tuple(messages)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _visible_html_text(value: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(value)
    parser.close()
    return "".join(parser.parts)


def _telegram_text_length(value: str) -> int:
    """Count UTF-16 code units after HTML entities are parsed, as Telegram does."""

    visible = _visible_html_text(value)
    return len(visible.encode("utf-16-le")) // 2


def _split_html_section(section: str, *, limit: int) -> tuple[str, ...]:
    if _telegram_text_length(section) <= limit:
        return (section,)
    fragments: list[str] = []
    current = ""
    for line in section.splitlines():
        if _telegram_text_length(line) > limit:
            if current:
                fragments.append(current)
                current = ""
            # Defensive fallback for unexpectedly large model-authored text. Normal
            # cards keep links; only one individually oversized line loses markup.
            visible = _visible_html_text(line)
            while visible:
                chunk = visible[:limit]
                while _telegram_text_length(chunk) > limit:
                    chunk = chunk[:-1]
                fragments.append(escape(chunk))
                visible = visible[len(chunk) :]
            continue
        candidate = line if not current else f"{current}\n{line}"
        if _telegram_text_length(candidate) <= limit:
            current = candidate
        else:
            fragments.append(current)
            current = line
    if current:
        fragments.append(current)
    return tuple(fragments)


def _format_proposal(
    index: int,
    recommendation: Recommendation,
    copy: ProposalCopy,
    *,
    checkout_items: Sequence[TripCheckoutItem] = (),
) -> str:
    proposal = recommendation.proposal
    combination = proposal.trip_option.combination
    destination = proposal.candidate.destination
    label = _recommendation_label(recommendation.labels)
    anchor_activities = _proposal_anchor_activities(proposal)
    city_url = build_yandex_maps_search_url(
        name=destination.name,
        region=destination.region,
    )
    lines = [
        f"<b>{index}. {_html_link(escape(copy.title), city_url)}</b>",
        f"{escape(label)}. {escape(copy.reason)}",
        _format_trip_length(combination),
        escape(destination.short_description or _fallback_city_description(proposal)),
        "Фото города — по ссылке в названии.",
    ]
    if anchor_activities:
        activity_line = "Коротко: " + "; ".join(
            _format_activity_link(item, destination) for item in anchor_activities
        )
        if not proposal.content_grounded:
            activity_line += " · идеи AI, детали пока не проверены"
        lines.append(activity_line)
    checkout_by_component = _checkout_items_by_component(
        checkout_items,
        proposal.trip_option,
        allow_unscoped=True,
    )
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
            _format_transport_total(combination),
            f"В городе: {_format_duration(combination.metrics.time_in_city)}",
        )
    )
    if combination.hotel is not None:
        lines.append(
            _format_hotel_booking_line(
                combination.hotel,
                destination_name=proposal.candidate.destination.name,
                destination_region=proposal.candidate.destination.region,
                checkout_item=checkout_by_component.get(TripComponent.HOTEL),
            )
        )
        lines.append(_format_hotel_summary(combination.hotel))
        if combination.hotel.price is not None and combination.hotel.currency is not None:
            lines.append(
                "Цена отеля: " + format_money(combination.hotel.price, combination.hotel.currency)
            )
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


def format_inspiration(
    recommendations: Sequence[Recommendation],
    copies: Sequence[ProposalCopy],
) -> str:
    copy_by_id = {item.proposal_id: item for item in copies}
    lines = [
        "<b>Вот куда можно уехать с вашим настроением</b>",
        "Я уже сверил примерную логистику и доступные цены через Tutu. Названия городов "
        "и мест открывают Яндекс Карты с фотографиями и отзывами.",
    ]
    for index, recommendation in enumerate(recommendations, start=1):
        proposal = recommendation.proposal
        destination = proposal.candidate.destination
        city_url = build_yandex_maps_search_url(
            name=destination.name,
            region=destination.region,
        )
        copy = copy_by_id[f"proposal_{index}"]
        activities = "; ".join(
            _format_activity_link(item, destination)
            for item in _proposal_anchor_activities(proposal)
        )
        lines.extend(
            (
                f"\n<b>{index}. {_html_link(escape(destination.name), city_url)}</b>",
                escape(destination.short_description or copy.reason),
                f"Что попробовать: {activities}",
                _format_total_cost(proposal),
            )
        )
    lines.append("\nЦены предварительные. Нажмите кнопку ниже — покажу билеты, отели и сравнение.")
    return "\n".join(lines)


def _format_activity_link(activity: Activity, destination=None) -> str:
    name = escape(activity.name)
    map_url = (
        str(activity.map_url)
        if activity.map_url is not None
        else build_yandex_maps_search_url(
            name=activity.name,
            address=activity.address,
            city=getattr(destination, "name", None),
            region=getattr(destination, "region", None),
        )
    )
    return f'<a href="{escape(map_url, quote=True)}">{name}</a>'


def _proposal_anchor_activities(proposal: WeekendProposal) -> tuple[Activity, ...]:
    candidates = [item.activity for day in proposal.days for item in day.activities]
    candidates.extend(item for day in proposal.days for item in day.suggestions)
    candidates.extend(proposal.suggested_activities)
    selected: list[Activity] = []
    seen_ids: set[str] = set()
    seen_places: set[str] = set()
    for activity in candidates:
        identities = activity_place_identities(activity)
        if activity.activity_id in seen_ids or seen_places.intersection(identities):
            continue
        selected.append(activity)
        seen_ids.add(activity.activity_id)
        seen_places.update(identities)
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


def _format_trip_length(combination: TripCombination) -> str:
    calendar_delta = (
        combination.return_offer.departure_at.date() - combination.outbound.departure_at.date()
    ).days
    days = max(
        calendar_delta + 1,
        1,
    )
    nights = combination.stay.nights if combination.stay is not None else 0
    return f"🗓 {days} дн. · {nights} ноч."


def _format_total_cost(proposal: WeekendProposal) -> str:
    if proposal.cost.confirmed_total is None or proposal.cost.confirmed_currency is None:
        return "💳 Предварительная стоимость поездки: пока неизвестна"
    return "💳 Предварительная стоимость поездки: " + format_money(
        proposal.cost.confirmed_total,
        proposal.cost.confirmed_currency,
    )


def _format_transport_total(combination: TripCombination) -> str:
    offers = (combination.outbound, combination.return_offer)
    currencies = {item.currency for item in offers if item.price is not None}
    if any(item.price is None for item in offers) or len(currencies) != 1:
        return "Стоимость транспорта: пока известна не полностью"
    currency = next(iter(currencies))
    total = sum((item.price for item in offers if item.price is not None), start=0)
    return f"Стоимость транспорта: {format_money(total, currency or '')}"


def _format_hotel_summary(hotel: HotelOffer) -> str:
    facts: list[str] = []
    if hotel.stars is not None:
        facts.append(f"{hotel.stars}★")
    if hotel.rating is not None:
        facts.append(f"рейтинг {hotel.rating}/10")
    if hotel.breakfast_included is True:
        facts.append("завтрак включён")
    if hotel.free_cancellation is True:
        facts.append("есть бесплатная отмена")
    return "Коротко об отеле: " + (", ".join(facts) if facts else "условия уточняются")


def _format_hotel_booking_line(
    hotel: HotelOffer,
    *,
    destination_name: str,
    destination_region: str,
    checkout_item: TripCheckoutItem | None = None,
) -> str:
    hotel_name = _linked_hotel_name(hotel, destination_name, destination_region)
    booking_url = str(checkout_item.link.url) if checkout_item is not None else None
    if hotel.room_name:
        room_name = escape(hotel.room_name)
        if booking_url is not None:
            room_name = _html_link(room_name, booking_url)
        booking = f" · номер: {room_name}"
    elif booking_url is not None:
        booking = f" · {_html_link('забронировать', booking_url)}"
    else:
        booking = ""
    return f"🏨 Отель: {hotel_name}{booking}"


def _format_hotel_details(
    hotel: HotelOffer,
    *,
    destination_name: str,
    destination_region: str,
    checkout_item: TripCheckoutItem | None = None,
) -> str:
    hotel_name = _linked_hotel_name(hotel, destination_name, destination_region)
    lines = [f"🏨 {hotel_name}", _format_hotel_summary(hotel)]
    if hotel.room_name:
        room_name = escape(hotel.room_name)
        if checkout_item is not None:
            room_name = _html_link(room_name, str(checkout_item.link.url))
        lines.append(f"Номер: {room_name}")
    if hotel.price is not None and hotel.currency is not None:
        lines.append(f"Цена: {format_money(hotel.price, hotel.currency)}")
    return "\n".join(lines)


def _format_hotel_plan(
    hotel: HotelOffer,
    *,
    destination_name: str,
    destination_region: str,
    checkout_item: TripCheckoutItem | None = None,
) -> str:
    check_in_time = hotel.check_in_time.strftime("%H:%M") if hotel.check_in_time else "уточнить"
    check_out_time = hotel.check_out_time.strftime("%H:%M") if hotel.check_out_time else "уточнить"
    summary = _format_hotel_summary(hotel).removeprefix("Коротко об отеле: ")
    hotel_name = _linked_hotel_name(hotel, destination_name, destination_region)
    booking = (
        f" · {_html_link('забронировать', str(checkout_item.link.url))}"
        if checkout_item is not None
        else ""
    )
    lines = [
        f"🏨 {hotel_name}{booking} · {summary}",
        f"Заезд: {hotel.check_in:%d.%m}, {check_in_time}",
        f"Выезд: {hotel.check_out:%d.%m}, {check_out_time}",
    ]
    if hotel.price is not None and hotel.currency is not None:
        lines.append(f"Цена отеля: {format_money(hotel.price, hotel.currency)}")
    return "\n".join(lines)


def _linked_hotel_name(hotel: HotelOffer, city: str, region: str) -> str:
    map_url = build_yandex_maps_search_url(
        name=hotel.name,
        city=city,
        region=region,
    )
    return _html_link(escape(hotel.name), map_url)


def _checkout_items_by_component(
    checkout_items: Sequence[TripCheckoutItem],
    option: RankedTripOption,
    *,
    allow_unscoped: bool,
) -> dict[TripComponent, TripCheckoutItem]:
    signature = option.combination.signature
    scoped = {item.component: item for item in checkout_items if item.option_signature == signature}
    if scoped or not allow_unscoped:
        return scoped
    return {item.component: item for item in checkout_items if item.option_signature is None}


def _fallback_city_description(proposal: WeekendProposal) -> str:
    names = [item.name for item in _proposal_anchor_activities(proposal)]
    if not names:
        return "Короткая поездка с программой под ваши интересы."
    return "Поездка вокруг мест: " + ", ".join(names) + "."


def _format_trip_comparison(proposal: WeekendProposal) -> str | None:
    alternatives = (proposal.trip_option, *proposal.trip_alternatives)
    all_with_price = [
        item for item in alternatives if item.combination.price.known_total is not None
    ]
    if not all_with_price:
        return None
    fastest = min(alternatives, key=lambda item: item.combination.metrics.total_travel_duration)
    cheapest = min(all_with_price, key=lambda item: item.combination.price.known_total)
    parts: list[str] = []
    for option, label in (
        (cheapest, "дешевле"),
        (fastest, "быстрее"),
    ):
        total = option.combination.price.known_total
        currency = option.combination.price.currency
        if total is not None:
            parts.append(f"{label} за {format_money(total, currency)}")
        else:
            parts.append(f"{label} — цена уточняется")
    if not parts:
        return None
    return "Сравнить варианты: " + "; ".join(parts)


def _format_expanded_alternative(
    option: RankedTripOption,
    all_options: tuple[RankedTripOption, ...],
    *,
    checkout_items: Sequence[TripCheckoutItem] = (),
    destination_name: str,
    destination_region: str,
) -> tuple[str, ...]:
    labels = []
    if SortPreference.CHEAPEST in option.labels:
        labels.append("бюджетный")
    if SortPreference.BALANCED in option.labels:
        labels.append("оптимальный")
    if SortPreference.FASTEST in option.labels:
        labels.append("быстрый")
    title = ", ".join(labels) if labels else "ещё один подходящий"
    combination = option.combination
    checkout_by_component = _checkout_items_by_component(
        checkout_items,
        option,
        allow_unscoped=False,
    )
    lines = [f"<b>{escape(title.capitalize())}</b>"]
    transport_labels = _transport_rank_labels(option, all_options=all_options)
    hotel_labels = _hotel_rank_labels(option, all_options=all_options)
    if transport_labels:
        lines.append(f"🚗 Транспорт: {', '.join(transport_labels)}")
    if combination.hotel is None:
        lines.append("🏨 Отель: не нужен")
    elif hotel_labels:
        lines.append(f"🏨 Категория отеля: {', '.join(hotel_labels)}")
    lines.append(
        _format_leg(
            "Туда",
            combination.outbound,
            checkout_by_component.get(TripComponent.OUTBOUND),
        )
    )
    lines.append(
        _format_leg(
            "Обратно",
            combination.return_offer,
            checkout_by_component.get(TripComponent.RETURN),
        )
    )
    if combination.hotel is not None:
        lines.append(
            _format_hotel_booking_line(
                combination.hotel,
                destination_name=destination_name,
                destination_region=destination_region,
                checkout_item=checkout_by_component.get(TripComponent.HOTEL),
            )
        )
        lines.append(_format_hotel_summary(combination.hotel))
    total = combination.price.known_total
    if total is not None:
        lines.append(
            "💳 Предварительная стоимость: " + format_money(total, combination.price.currency)
        )
    return tuple(lines)


def _transport_rank_labels(
    option: RankedTripOption,
    *,
    all_options: tuple[RankedTripOption, ...],
) -> tuple[str, ...]:
    labels = []
    ranked = [
        candidate
        for candidate in all_options
        if candidate.combination.price.known_total is not None
    ]
    if not ranked:
        return ()
    transport_cheapest = min(
        ranked,
        key=lambda item: item.combination.price.known_total or Decimal("Infinity"),
    )
    transport_fastest = min(ranked, key=lambda item: item.combination.metrics.total_travel_duration)
    if option == transport_cheapest:
        labels.append("бюджетный")
    if option == transport_fastest:
        labels.append("быстрый")
    if SortPreference.BALANCED in option.labels and "оптимальный" not in labels:
        labels.append("оптимальный")
    return tuple(labels)


def _hotel_rank_labels(
    option: RankedTripOption,
    *,
    all_options: tuple[RankedTripOption, ...],
) -> tuple[str, ...]:
    labels = []
    ranked_hotel = [
        candidate
        for candidate in all_options
        if candidate.combination.hotel is not None and candidate.combination.hotel.price is not None
    ]
    if option.combination.hotel is None:
        return ("не нужен",)
    if not ranked_hotel:
        if SortPreference.BALANCED in option.labels:
            labels.append("оптимальный")
        return tuple(labels)
    hotel_cheapest = min(
        ranked_hotel,
        key=lambda item: (
            item.combination.hotel.price if item.combination.hotel else Decimal("Infinity")
        ),
    )
    if option.combination.hotel == hotel_cheapest.combination.hotel:
        labels.append("бюджетный")
    if SortPreference.BALANCED in option.labels:
        labels.append("оптимальный")
    return tuple(dict.fromkeys(labels))
