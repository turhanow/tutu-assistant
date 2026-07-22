"""Conservative recovery for known-route fields and explicit contradictions."""

from __future__ import annotations

import re
from datetime import date

from app.domain.models import ParsedTripDraft

_CITY_ALIASES = {
    "москва": "Москва",
    "москвы": "Москва",
    "санкт-петербург": "Санкт-Петербург",
    "санкт-петербурга": "Санкт-Петербург",
    "питер": "Санкт-Петербург",
    "питера": "Санкт-Петербург",
    "казань": "Казань",
    "казани": "Казань",
    "ярославль": "Ярославль",
    "ярославля": "Ярославль",
}
_MONTHS = {
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
_DATE = r"(?P<day>\d{1,2})\s+(?P<month>" + "|".join(_MONTHS) + r")(?:\s+(?P<year>20\d{2}))?"


def normalize_city_answer(value: str) -> str:
    """Remove conversational prepositions and punctuation from a form city answer."""

    normalized = " ".join(value.strip().strip(" .,!?:;").split())
    normalized = re.sub(r"^(?:из|в)\s+(?:города\s+)?", "", normalized, flags=re.IGNORECASE)
    if (
        not normalized
        or len(normalized) > 200
        or not re.fullmatch(r"[A-Za-zА-Яа-яЁё -]+", normalized)
    ):
        raise ValueError("Введите только название города, например «Москва».")
    return _CITY_ALIASES.get(normalized.casefold().replace("ё", "е"), normalized)


def explicit_conflicts(text: str) -> tuple[str, ...]:
    """Name only contradictions that are explicit in the same user message."""

    normalized = " ".join(text.casefold().replace("ё", "е").split())
    conflicts: list[str] = []
    hotel_required = bool(re.search(r"\b(?:нужен отель|отель нужен|с отелем)\b", normalized))
    hotel_forbidden = bool(
        re.search(
            r"\b(?:без отеля|отель не нужен|не нужен отель|отель нужен и не нужен)\b",
            normalized,
        )
    )
    if hotel_required and hotel_forbidden:
        conflicts.append("одновременно указано, что отель нужен и не нужен")
    only_flight = bool(re.search(r"\b(?:только самолет|только авиа)\w*\b", normalized))
    no_flight = bool(
        re.search(r"\b(?:самолет\w* не предлагать|без самолет\w*|не лететь)\b", normalized)
    )
    if only_flight and no_flight:
        conflicts.append("самолёт одновременно обязателен и запрещён")
    return tuple(conflicts)


def recover_known_draft(text: str, *, today: date) -> ParsedTripDraft:
    """Recover route and explicitly labelled dates without inventing missing values."""

    origin = destination = None
    route = re.search(
        r"(?P<origin>[А-ЯЁ][А-Яа-яЁё -]{1,60}?)\s*[—–-]\s*"
        r"(?P<destination>[А-ЯЁ][А-Яа-яЁё -]{1,60}?)(?=,|$)",
        text,
    )
    if route is not None:
        origin = normalize_city_answer(route.group("origin"))
        destination = normalize_city_answer(route.group("destination"))

    explicit_year = re.search(r"\b(20\d{2})\b", text)
    default_year = int(explicit_year.group(1)) if explicit_year else today.year
    departure = _labelled_date(text, "туда", default_year)
    returning = _labelled_date(text, "обратно", default_year)
    return ParsedTripDraft(
        origin=origin,
        destination=destination,
        departure_date=departure,
        return_date=returning,
    )


def _labelled_date(text: str, label: str, default_year: int) -> date | None:
    match = re.search(rf"\b{label}\s+{_DATE}", text.casefold().replace("ё", "е"))
    if match is None:
        return None
    return date(
        int(match.group("year") or default_year),
        _MONTHS[match.group("month")],
        int(match.group("day")),
    )
