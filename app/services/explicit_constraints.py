"""Conservative deterministic guardrails for constraints that must never be lost by an LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from app.domain.models import HotelMode

_NUMBER_WORDS = {
    "один": 1,
    "одна": 1,
    "одно": 1,
    "двое": 2,
    "два": 2,
    "две": 2,
    "трое": 3,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}
_NUMBER = r"(?:\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")"


@dataclass(frozen=True, slots=True)
class ExplicitParty:
    adults: int | None = None
    children: int | None = None
    rooms: int | None = None


def normalize_user_text(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def extract_explicit_party(text: str) -> ExplicitParty:
    normalized = normalize_user_text(text)
    adults = _count_before(normalized, r"взросл\w*")
    children = _count_before(normalized, r"(?:ребен\w*|дет\w*)")
    rooms = _count_before(normalized, r"номер\w*")
    if children is None and re.search(r"\b(?:с ребенком|ребенок|ребенка|дети)\b", normalized):
        children = 1
    return ExplicitParty(adults=adults, children=children, rooms=rooms)


def explicit_hotel_mode(text: str) -> HotelMode | None:
    normalized = normalize_user_text(text)
    if re.search(r"\b(?:без отеля|отель не нужен|не нужен отель)\b", normalized):
        return HotelMode.FORBIDDEN
    if re.search(r"\b(?:отель необязателен|отель не обязателен|можно без отеля)\b", normalized):
        return HotelMode.OPTIONAL
    if re.search(r"\b(?:нужен отель|отель нужен|с отелем)\b", normalized):
        return HotelMode.REQUIRED
    return None


def resolved_hotel_mode(
    text: str,
    extracted: HotelMode | None,
    *,
    departure_date: date | None,
    return_date: date | None,
) -> HotelMode | None:
    """Prefer explicit wording, then require a hotel for an overnight trip by default."""

    explicit = explicit_hotel_mode(text)
    if explicit is not None:
        return explicit
    if departure_date is not None and return_date is not None and return_date > departure_date:
        return HotelMode.REQUIRED
    return extracted


def explicitly_relaxed(text: str) -> bool:
    normalized = normalize_user_text(text)
    return bool(re.search(r"\b(?:без суеты|не спеша|спокойн\w*)\b", normalized))


def explicitly_forbids_night_travel(text: str) -> bool:
    normalized = normalize_user_text(text)
    return bool(
        re.search(
            r"\b(?:без ночн\w* (?:дорог\w*|переезд\w*|поезд\w*)|не ехать ночью)\b",
            normalized,
        )
    )


def explicitly_mentions_river(text: str) -> bool:
    normalized = normalize_user_text(text)
    return bool(re.search(r"\b(?:рек\w*|набережн\w*|у воды)\b", normalized))


def _count_before(text: str, noun_pattern: str) -> int | None:
    match = re.search(rf"\b(?P<count>{_NUMBER})\s+{noun_pattern}\b", text)
    if match is None:
        return None
    value = match.group("count")
    return int(value) if value.isdigit() else _NUMBER_WORDS[value]
