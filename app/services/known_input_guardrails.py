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
_MONTH_PATTERN = "|".join(_MONTHS)


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

    departure, returning = extract_explicit_trip_dates(text, today=today)
    return ParsedTripDraft(
        origin=origin,
        destination=destination,
        departure_date=departure,
        return_date=returning,
    )


def extract_explicit_trip_dates(
    text: str,
    *,
    today: date,
) -> tuple[date | None, date | None]:
    """Prefer labelled leg dates, then an explicit range in its written order."""

    explicit_year = re.search(r"\b(20\d{2})\b", text)
    default_year = int(explicit_year.group(1)) if explicit_year else today.year
    departure = _labelled_date(text, "туда", default_year)
    returning = _labelled_date(text, "обратно", default_year)
    explicit_range = extract_explicit_date_range(text, today=today)
    if explicit_range is not None and departure is None and returning is None:
        return explicit_range
    return departure, returning


def extract_explicit_date_range(text: str, *, today: date) -> tuple[date, date] | None:
    """Extract an explicit Russian date range without asking an LLM to reorder it."""

    normalized = " ".join(text.casefold().replace("ё", "е").split())
    same_month = re.search(
        rf"(?<!\d)(?P<first>\d{{1,2}})\s*[—–-]\s*(?P<second>\d{{1,2}})\s+"
        rf"(?P<month>{_MONTH_PATTERN})(?:\s+(?P<year>20\d{{2}}))?",
        normalized,
    )
    if same_month is not None:
        year = int(same_month.group("year") or today.year)
        month = _MONTHS[same_month.group("month")]
        departure = _safe_date(year, month, int(same_month.group("first")))
        returning = _safe_date(year, month, int(same_month.group("second")))
        if departure is None or returning is None:
            return None
        if same_month.group("year") is None and departure < today:
            departure = departure.replace(year=year + 1)
            returning = returning.replace(year=year + 1)
        return departure, returning

    numeric = re.search(
        r"(?<!\d)(?P<first>\d{1,2})\s*[—–-]\s*(?P<second>\d{1,2})[./]"
        r"(?P<month>\d{1,2})(?:[./](?P<year>20\d{2}))?(?!\d)",
        normalized,
    )
    if numeric is not None:
        year = int(numeric.group("year") or today.year)
        month = int(numeric.group("month"))
        departure = _safe_date(year, month, int(numeric.group("first")))
        returning = _safe_date(year, month, int(numeric.group("second")))
        if departure is None or returning is None:
            return None
        if numeric.group("year") is None and departure < today:
            departure = departure.replace(year=year + 1)
            returning = returning.replace(year=year + 1)
        return departure, returning

    full_range = re.search(
        rf"(?<!\d)(?P<first>\d{{1,2}})\s+(?P<first_month>{_MONTH_PATTERN})"
        rf"(?:\s+(?P<first_year>20\d{{2}}))?\s*[—–-]\s*"
        rf"(?P<second>\d{{1,2}})\s+(?P<second_month>{_MONTH_PATTERN})"
        rf"(?:\s+(?P<second_year>20\d{{2}}))?",
        normalized,
    )
    if full_range is None:
        return None
    first_year = int(full_range.group("first_year") or today.year)
    first_month = _MONTHS[full_range.group("first_month")]
    second_month = _MONTHS[full_range.group("second_month")]
    departure = _safe_date(first_year, first_month, int(full_range.group("first")))
    if departure is None:
        return None
    if full_range.group("first_year") is None and departure < today:
        first_year += 1
        departure = departure.replace(year=first_year)
    second_year = int(full_range.group("second_year") or first_year)
    if full_range.group("second_year") is None and second_month < first_month:
        second_year += 1
    returning = _safe_date(second_year, second_month, int(full_range.group("second")))
    return (departure, returning) if returning is not None else None


def _labelled_date(text: str, label: str, default_year: int) -> date | None:
    match = re.search(rf"\b{label}\s+{_DATE}", text.casefold().replace("ё", "е"))
    if match is None:
        return None
    return date(
        int(match.group("year") or default_year),
        _MONTHS[match.group("month")],
        int(match.group("day")),
    )


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
