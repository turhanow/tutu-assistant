"""HTML-safe presentation for destination discovery and verified proposals."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from html import escape

from app.bot.formatters import TRANSPORT_LABELS, format_money
from app.domain.discovery_models import (
    CandidateShortlist,
    ProposalCopy,
    Recommendation,
)

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
        return "<b>Подходящих направлений пока нет</b>\nПопробуйте изменить интересы или даты."
    lines = ["<b>Короткий список направлений</b>"]
    for index, item in enumerate(shortlist.candidates, start=1):
        reasons = "; ".join(escape(reason) for reason in item.match_reasons)
        lines.append(f"{index}. <b>{escape(item.destination.name)}</b> — {reasons}")
    lines.append("Теперь проверяю транспорт, проживание и реальное время в городе.")
    return "\n".join(lines)


def format_proposal_sections(
    recommendations: Sequence[Recommendation],
    copies: Sequence[ProposalCopy],
) -> tuple[str, ...]:
    copy_by_id = {item.proposal_id: item for item in copies}
    sections = [
        "<b>Проверенные идеи для поездки</b>\nЦена и доступность актуальны на момент поиска."
    ]
    for index, recommendation in enumerate(recommendations, start=1):
        copy = copy_by_id[f"proposal_{index}"]
        sections.append(_format_proposal(index, recommendation, copy))
    return tuple(sections)


def format_program(recommendation: Recommendation) -> str:
    proposal = recommendation.proposal
    lines = [f"<b>Программа: {escape(proposal.candidate.destination.name)}</b>"]
    for day_index, day in enumerate(proposal.days, start=1):
        lines.append(f"\n<b>День {day_index} · {day.date:%d.%m}</b>")
        if not day.activities:
            lines.append("Свободное время — проверенные активности не размещены.")
            continue
        for item in day.activities:
            lines.append(
                f"{item.starts_at:%H:%M}–{item.ends_at:%H:%M} · {escape(item.activity.name)}"
            )
    combination = proposal.trip_option.combination
    lines.extend(
        (
            "\n<b>Логистика</b>",
            _format_leg("Туда", combination.outbound),
            _format_leg("Обратно", combination.return_offer),
            f"В городе: {_format_duration(combination.metrics.time_in_city)}",
            f"Проверено: {proposal.verified_at:%d.%m.%Y %H:%M}",
            f"\n⚠️ {escape(proposal.trade_off)}",
        )
    )
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
) -> str:
    proposal = recommendation.proposal
    combination = proposal.trip_option.combination
    label = _recommendation_label(recommendation.labels)
    lines = [
        f"<b>{index}. {escape(copy.title)}</b>",
        f"{escape(label)}. {escape(copy.reason)}",
        _format_leg("Туда", combination.outbound),
        _format_leg("Обратно", combination.return_offer),
        f"В дороге: {_format_duration(combination.metrics.total_travel_duration)}",
        f"В городе: {_format_duration(combination.metrics.time_in_city)}",
    ]
    if combination.hotel is not None:
        lines.append(f"Отель: {escape(combination.hotel.name)}")
    else:
        lines.append("Отель: не нужен")
    cost = proposal.cost
    if cost.confirmed_total is not None and cost.confirmed_currency is not None:
        lines.append(
            "Подтверждённая стоимость: "
            + format_money(cost.confirmed_total, cost.confirmed_currency)
        )
    else:
        lines.append("Подтверждённая стоимость: пока неизвестна")
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
    anchor_names = [item.activity.name for day in proposal.days for item in day.activities][:2]
    if anchor_names:
        lines.append("Что делать: " + "; ".join(escape(item) for item in anchor_names))
    lines.append(f"⚠️ {escape(copy.trade_off)}")
    return "\n".join(lines)


def _recommendation_label(labels: frozenset[str]) -> str:
    ordered = [
        text
        for key, text in (
            ("best_match", "лучшее общее соответствие"),
            ("lowest_confirmed_cost", "самая низкая подтверждённая стоимость"),
            ("most_time_in_city", "больше всего полезного времени в городе"),
        )
        if key in labels
    ]
    if len(ordered) > 1:
        return "Лидер по нескольким критериям: " + ", ".join(ordered)
    return ordered[0].capitalize() if ordered else "Проверенный вариант"


def _format_leg(label, offer) -> str:
    transfers = (
        "пересадки не указаны"
        if offer.transfers is None
        else "без пересадок"
        if offer.transfers == 0
        else f"пересадок: {offer.transfers}"
    )
    return (
        f"{label}: {TRANSPORT_LABELS[offer.mode]} · "
        f"{_format_datetime(offer.departure_at)} → {_format_datetime(offer.arrival_at)} · "
        f"{transfers}"
    )


def _format_datetime(value: datetime) -> str:
    return f"{value:%d.%m, %H:%M}"


def _format_duration(value: timedelta) -> str:
    total_minutes = max(int(value.total_seconds() // 60), 0)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours} ч {minutes} мин" if minutes else f"{hours} ч"
