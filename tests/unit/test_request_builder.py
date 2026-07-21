from datetime import date

import pytest

from app.domain.models import HotelMode, ParsedTripDraft, SortPreference, TransportMode
from app.services.request_builder import (
    DraftInputError,
    apply_form_answer,
    build_trip_request,
    next_missing_field,
)


def test_structured_form_completes_draft_deterministically() -> None:
    draft = ParsedTripDraft()
    values = {
        "origin": "Москва",
        "destination": "Казань",
        "departure_date": "21.08.2026",
        "return_date": "23.08.2026",
        "hotel_mode": "да",
    }
    while (field := next_missing_field(draft)) is not None:
        draft = apply_form_answer(draft, field, values[field])

    request = build_trip_request(draft)

    assert request.departure_date == date(2026, 8, 21)
    assert request.hotel.mode is HotelMode.REQUIRED


@pytest.mark.parametrize("value", ["завтра", "31.02.2026", "2026/08/21"])
def test_form_rejects_ambiguous_or_invalid_dates(value: str) -> None:
    with pytest.raises(DraftInputError, match=r"ДД\.ММ\.ГГГГ"):
        apply_form_answer(ParsedTripDraft(), "departure_date", value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("завтра", date(2026, 7, 22)),
        ("25 июля", date(2026, 7, 25)),
        ("20 июля", date(2027, 7, 20)),
        ("25 июля 2026", date(2026, 7, 25)),
    ],
)
def test_form_accepts_human_dates_with_explicit_clock(value: str, expected: date) -> None:
    result = apply_form_answer(
        ParsedTripDraft(),
        "departure_date",
        value,
        today=date(2026, 7, 21),
    )

    assert result.departure_date == expected


def test_build_error_identifies_only_the_field_that_needs_correction() -> None:
    draft = ParsedTripDraft(
        origin="Москва",
        destination="Москва",
        departure_date="2026-08-21",
        return_date="2026-08-23",
        hotel_mode=HotelMode.FORBIDDEN,
    )

    with pytest.raises(DraftInputError) as raised:
        build_trip_request(draft)

    assert raised.value.field == "destination"


def test_past_departure_is_rejected_before_confirmation() -> None:
    draft = ParsedTripDraft(
        origin="Москва",
        destination="Казань",
        departure_date="2026-07-15",
        return_date="2026-07-17",
        hotel_mode=HotelMode.FORBIDDEN,
    )

    with pytest.raises(DraftInputError, match="уже прошла") as raised:
        build_trip_request(draft, today=date(2026, 7, 22))

    assert raised.value.field == "departure_date"


def test_same_day_required_hotel_is_rejected_before_confirmation() -> None:
    draft = ParsedTripDraft(
        origin="Москва",
        destination="Тула",
        departure_date="2026-08-22",
        return_date="2026-08-22",
        hotel_mode=HotelMode.REQUIRED,
    )

    with pytest.raises(DraftInputError, match="нет ночи") as raised:
        build_trip_request(draft, today=date(2026, 7, 22))

    assert raised.value.field == "hotel_mode"


@pytest.mark.parametrize(
    "draft",
    [
        ParsedTripDraft(adults=3),
        ParsedTripDraft(adults=2, children=1),
        ParsedTripDraft(adults=2, rooms=2),
    ],
)
def test_unsupported_party_is_explained_without_silent_truncation(draft) -> None:
    complete = draft.model_copy(
        update={
            "origin": "Москва",
            "destination": "Казань",
            "departure_date": date(2026, 8, 15),
            "return_date": date(2026, 8, 17),
            "hotel_mode": HotelMode.REQUIRED,
        }
    )

    with pytest.raises(DraftInputError, match="без детей и один номер") as raised:
        build_trip_request(complete, today=date(2026, 7, 22))

    assert raised.value.field == "adults"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Да, нужен отель.", HotelMode.REQUIRED),
        ("Нет, отель не нужен.", HotelMode.FORBIDDEN),
        ("Отель необязателен.", HotelMode.OPTIONAL),
    ],
)
def test_hotel_form_accepts_natural_short_answers(answer: str, expected: HotelMode) -> None:
    result = apply_form_answer(ParsedTripDraft(), "hotel_mode", answer)

    assert result.hotel_mode is expected


@pytest.mark.parametrize(
    ("field", "value", "attribute", "expected"),
    [
        ("hotel_mode", "не нужен", "hotel_mode", HotelMode.FORBIDDEN),
        ("hotel_mode", "необязательно", "hotel_mode", HotelMode.OPTIONAL),
        ("adults", "2", "adults", 2),
        ("budget", "12 500,50", "budget", 12500.5),
        ("budget", "нет", "budget", None),
        (
            "allowed_modes",
            "поезд и автобус",
            "allowed_modes",
            {TransportMode.RAIL, TransportMode.BUS},
        ),
        ("allowed_modes", "любые", "allowed_modes", set()),
        ("sort", "хочу дешевле", "sort", SortPreference.CHEAPEST),
        ("sort", "меньше времени", "sort", SortPreference.FASTEST),
        ("sort", "баланс и удобство", "sort", SortPreference.BALANCED),
    ],
)
def test_editable_fields_have_deterministic_form_mapping(
    field: str,
    value: str,
    attribute: str,
    expected,
) -> None:
    result = apply_form_answer(ParsedTripDraft(), field, value)

    actual = getattr(result, attribute)
    if attribute == "budget" and actual is not None:
        actual = float(actual)
    assert actual == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hotel_mode", "может быть"),
        ("adults", "3"),
        ("budget", "дорого"),
        ("budget", "-1"),
        ("allowed_modes", "телепорт"),
        ("sort", "красивее"),
        ("unsupported", "value"),
    ],
)
def test_invalid_form_edits_return_actionable_error(field: str, value: str) -> None:
    with pytest.raises(DraftInputError):
        apply_form_answer(ParsedTripDraft(), field, value)
